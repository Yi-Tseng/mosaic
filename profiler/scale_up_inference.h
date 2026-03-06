// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef SCALE_UP_INFERENCE_H_
#define SCALE_UP_INFERENCE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define SCALE_UP_MAX_TRANSFER_BYTES    (1024 * 1024)  // 1 MB
#define SCALE_UP_SLICE_SPLIT_THRESHOLD (64 * 1024)    // 64 KB
#define SCALE_UP_TREE_TRANSFER_BYTES   (32 * 1024)    // 32 KB - Tree algorithm fixed chunk size

/**
 * Result of transfer inference for a single collective or P2P operation.
 */
struct InferredTransfers
{
    size_t perTransferBytes;     // Individual transfer size (capped at 1 MB)
    int numTransfers;            // Total number of transfers for this rank
    size_t totalRankBytes;       // Total bytes this rank transfers through the internal network
    double networkTimeFraction;  // Fraction of collective time assumed to be networking [0.0, 1.0]
    int stepsPerRank;            // Number of logical steps this rank participates in
    int numChannels;             // Number of channels used
};

/**
 * @brief Infer transfer characteristics for a collective operation on scale-up (no proxy ops).
 *
 * Uses the collective type, data size, rank count and channel count to estimate:
 * - Per-transfer size (capped at 1 MB, subdivided if larger)
 * - Total number of transfers for this rank
 * - Total bytes transferred by this rank on the internal network
 *
 * Supported collectives: AllReduce, AllGather, ReduceScatter, Broadcast, Reduce.
 * Unknown collectives fall back to a single-transfer model.
 *
 * IMPORTANT: The profiler event's collectiveBytes (= count * typeSize) represents
 * different quantities depending on the collective type due to NCCL API conventions:
 *   - AllReduce:     count = total elements      → collectiveBytes = D (total data)
 *   - AllGather:     count = per-rank sendcount   → collectiveBytes = D/N
 *   - ReduceScatter: count = per-rank recvcount   → collectiveBytes = D/N
 *   - Broadcast:     count = total elements       → collectiveBytes = D (total data)
 *   - Reduce:        count = total elements       → collectiveBytes = D (total data)
 *
 * NCCL internally always works with nBytes = D (the global data size).
 * The ring algorithm divides nBytes into nRanks * nChannels chunks, so the
 * per-step per-channel transfer size is always D / (nRanks * nChannels).
 *
 * @param[in] func            Collective function name (e.g. "AllReduce").
 * @param[in] algo            Algorithm name (e.g. "Ring", "Tree"). NULL defaults to Ring.
 * @param[in] collectiveBytes Data size from profiler event (count * datatypeSize).
 * @param[in] nRanks          Number of ranks in the communicator.
 * @param[in] nChannels       Number of channels used by this collective.
 * @param[in] networkPct      Percentage of collective time assumed spent on networking (1-100).
 *
 * @return InferredTransfers with the estimated transfer parameters.
 */
inline InferredTransfers inferCollectiveTransfers(const char* func, const char* algo, size_t collectiveBytes,
                                                  int nRanks, uint8_t nChannels, double networkPct)
{
    InferredTransfers result   = {};
    result.networkTimeFraction = (networkPct > 0 && networkPct <= 100) ? networkPct / 100.0 : 1.0;
    result.numChannels         = nChannels > 0 ? nChannels : 1;

    if (collectiveBytes == 0 || nRanks <= 1)
    {
        result.perTransferBytes = 0;
        result.numTransfers     = 0;
        result.totalRankBytes   = 0;
        result.stepsPerRank     = 0;
        return result;
    }

    // Compute nBytesGlobal: the total data size that NCCL works with internally.
    // For AllGather/ReduceScatter, the profiler event's count is per-rank, so we
    // must multiply by nRanks to get the global size (see ncclFuncMaxSendRecvCount
    // in nccl/src/include/enqueue.h).
    //
    // Also determine steps per rank from NCCL's ring algorithm (from nccl/src/enqueue.cc):
    //   ncclPatternRingTwice (AllReduce): nstepsPerLoop = 2*(nRanks-1)
    //   ncclPatternRing (AllGather/ReduceScatter): nstepsPerLoop = nRanks-1
    //
    // And trafficMultiplier for totalRankBytes calculation.
    size_t nBytesGlobal      = collectiveBytes;
    double trafficMultiplier = 1.0;
    int stepsPerRank         = 1;

    if (func)
    {
        if (strstr(func, "AllReduce"))
        {
            nBytesGlobal      = collectiveBytes;  // count = total
            trafficMultiplier = 2.0;
            stepsPerRank      = 2 * (nRanks - 1);
        }
        else if (strstr(func, "AllGather"))
        {
            nBytesGlobal      = collectiveBytes * (size_t)nRanks;  // count = per-rank sendcount
            trafficMultiplier = (double)nRanks;
            stepsPerRank      = nRanks - 1;
        }
        else if (strstr(func, "ReduceScatter"))
        {
            nBytesGlobal      = collectiveBytes * (size_t)nRanks;  // count = per-rank recvcount
            trafficMultiplier = (double)nRanks;
            stepsPerRank      = nRanks - 1;
        }
        else if (strstr(func, "Broadcast") || strstr(func, "Reduce"))
        {
            nBytesGlobal      = collectiveBytes;  // count = total
            trafficMultiplier = 1.0;
            stepsPerRank      = 1;
        }
    }

    result.stepsPerRank = stepsPerRank;

    // Total bytes this rank transfers = collectiveBytes * trafficMultiplier * (nRanks-1) / nRanks.
    // This formula works uniformly because trafficMultiplier compensates for the
    // per-rank vs total count convention:
    //   AllReduce:     D * 2 * (N-1)/N = 2*(N-1)*D/N
    //   AllGather:     (D/N) * N * (N-1)/N = (N-1) * D/N
    //   ReduceScatter: (D/N) * N * (N-1)/N = (N-1) * D/N
    result.totalRankBytes =
        (size_t)((double)collectiveBytes * trafficMultiplier * (double)(nRanks - 1) / (double)nRanks);

    // Check if this is a Tree-based algorithm.
    // Tree does NOT divide data by nRanks like Ring. Instead, it sends data through
    // the tree hierarchy with per-transfer size capped at 32KB (chunkSize floor from
    // nccl/src/enqueue.cc). For Tree: chunkSteps=1, sliceSteps=1, nstepsPerLoop=1,
    // nchunksPerLoop=1. The per-channel transfer size is nBytesGlobal / nChannels,
    // then capped at 32KB.
    bool isTree = algo && (strstr(algo, "TREE") || strstr(algo, "Tree"));

    if (isTree)
    {
        size_t treePerChannel = nBytesGlobal / (size_t)result.numChannels;
        if (treePerChannel == 0) treePerChannel = 1;

        size_t treeTransferSize =
            treePerChannel <= SCALE_UP_TREE_TRANSFER_BYTES ? treePerChannel : SCALE_UP_TREE_TRANSFER_BYTES;

        result.perTransferBytes = treeTransferSize;
        result.numTransfers =
            result.totalRankBytes > 0 ? (int)std::ceil((double)result.totalRankBytes / treeTransferSize) : 0;
    }
    else
    {
        // Ring algorithm: divide the global data into nRanks chunks across nChannels.
        // From nccl/src/enqueue.cc:2234: loopSize = nChannels * nchunksPerLoop * chunkSize
        // where nchunksPerLoop = nRanks and chunkSize = nBytesGlobal / (nRanks * nChannels).
        size_t baseTransferSize = nBytesGlobal / (size_t)nRanks / (size_t)result.numChannels;
        if (baseTransferSize == 0) baseTransferSize = 1;

        // NCCL SIMPLE protocol ring uses SlicePerChunk = chunkSteps/sliceSteps = 2
        // for AllReduce, AllGather, ReduceScatter (see nccl/src/include/collectives.h).
        // Each ring step's chunk is sent in 2 slices, so the actual per-transfer size
        // is chunkSize/2. This applies when the per-step size is >= 64KB.
        int slicesPerChunk = 1;
        if (baseTransferSize >= SCALE_UP_SLICE_SPLIT_THRESHOLD)
        {
            slicesPerChunk = 2;
            baseTransferSize /= 2;
        }

        // Apply 1 MB cap: if a single slice transfer exceeds 1 MB, subdivide
        int numSubTransfers = 1;
        size_t perTransfer  = baseTransferSize;
        if (baseTransferSize > SCALE_UP_MAX_TRANSFER_BYTES)
        {
            numSubTransfers = (int)std::ceil((double)baseTransferSize / SCALE_UP_MAX_TRANSFER_BYTES);
            perTransfer     = SCALE_UP_MAX_TRANSFER_BYTES;
        }

        result.perTransferBytes = perTransfer;
        result.numTransfers     = stepsPerRank * result.numChannels * slicesPerChunk * numSubTransfers;
    }

    return result;
}

/**
 * @brief Infer transfer characteristics for a P2P operation on scale-up.
 *
 * For P2P (Send), the entire message is one logical transfer divided across channels,
 * with each channel transfer capped at 1 MB.
 *
 * @param[in] p2pBytes    Total bytes in the P2P operation.
 * @param[in] nChannels   Number of channels used.
 * @param[in] networkPct  Percentage of P2P time assumed spent on networking (1-100).
 *
 * @return InferredTransfers with the estimated transfer parameters.
 */
inline InferredTransfers inferP2PTransfers(size_t p2pBytes, uint8_t nChannels, double networkPct)
{
    InferredTransfers result   = {};
    result.networkTimeFraction = (networkPct > 0 && networkPct <= 100) ? networkPct / 100.0 : 1.0;
    result.numChannels         = nChannels > 0 ? nChannels : 1;
    result.stepsPerRank        = 1;

    if (p2pBytes == 0)
    {
        result.perTransferBytes = 0;
        result.numTransfers     = 0;
        result.totalRankBytes   = 0;
        return result;
    }

    size_t perChannelBytes = p2pBytes / (size_t)result.numChannels;
    if (perChannelBytes == 0) perChannelBytes = 1;

    // Apply the same slice split adjustment as for collectives: when the
    // per-channel transfer is >= 64KB, the actual per-transfer size is halved.
    int slicesPerChunk = 1;
    if (perChannelBytes >= SCALE_UP_SLICE_SPLIT_THRESHOLD)
    {
        slicesPerChunk = 2;
        perChannelBytes /= 2;
    }

    int numSubTransfers = 1;
    size_t perTransfer  = perChannelBytes;
    if (perChannelBytes > SCALE_UP_MAX_TRANSFER_BYTES)
    {
        numSubTransfers = (int)std::ceil((double)perChannelBytes / SCALE_UP_MAX_TRANSFER_BYTES);
        perTransfer     = SCALE_UP_MAX_TRANSFER_BYTES;
    }

    result.perTransferBytes = perTransfer;
    result.numTransfers     = result.numChannels * slicesPerChunk * numSubTransfers;
    result.totalRankBytes   = p2pBytes;

    return result;
}

#endif  // SCALE_UP_INFERENCE_H_
