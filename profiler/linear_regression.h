// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef LINEAR_REGRESSION_H_
#define LINEAR_REGRESSION_H_

#include <map>
#include <string>
#include <utility>  // For std::pair
#include <vector>

/**
 * Simple linear regression class for computing latency from transfer data.
 *
 * Given pairs of (size, time), computes:
 * - Slope: transfer time per byte (us/byte)
 * - Intercept: latency at size=0 (in us) - represents fixed overhead/latency
 *
 * Primary use case: The intercept from linear regression is used to estimate the
 * fixed latency component in rank-to-rank transfers. Rate/bandwidth is now computed
 * using interval-based active time calculation instead of 1/slope.
 *
 * Supports two modes:
 * - AVG: Use all transfer data points for regression
 * - MIN: Use minimum transfer time for each unique transfer size
 *
 * NOTE: This class is NOT thread-safe. Each instance should be used by a single thread.
 */
class LinearRegression
{
public:
    enum class Mode
    {
        AVG,  // Use all data points
        MIN   // Use minimum time per size
    };

    /**
     * @brief Construct a LinearRegression instance.
     *
     * @param[in] mode Regression mode: AVG (use all points) or MIN (use min time per size).
     */
    LinearRegression(Mode mode = Mode::AVG);

    /**
     * @brief Add a data point (size, time) to the regression.
     *
     * @param[in] x Transfer size (typically in bytes).
     * @param[in] y Transfer time (typically in microseconds).
     *
     * @note For MIN mode, stores the minimum time for each unique size.
     */
    void addPoint(double x, double y);

    /**
     * @brief Merge data from another LinearRegression instance.
     *
     * Combines data points from another instance into this one. Useful for
     * aggregating data across multiple windows or threads.
     *
     * @param[in] other Another LinearRegression instance to merge.
     *
     * @note For MIN mode, takes the minimum time for each size.
     */
    void merge(const LinearRegression& other);

    /**
     * @brief Clear all data points and reset statistics.
     */
    void clear();

    /**
     * @brief Calculate linear regression slope and intercept.
     *
     * Computes y = slope * x + intercept using least squares method.
     * With x = bytes and y = time (us): slope is transfer time per byte (us/byte)
     * and intercept is the fixed latency overhead (us).
     *
     * @param[out] slope Calculated slope (us/byte — transfer time per byte).
     * @param[out] intercept Calculated intercept (us — fixed latency at size 0).
     *
     * @return true if calculation succeeded (requires at least 2 points), false otherwise.
     */
    bool calculate(double& slope, double& intercept) const;

    /**
     * @brief Check if there are at least 3 different transfer sizes.
     *
     * Used to validate that regression has sufficient data diversity.
     *
     * @return true if at least 3 different sizes exist, false otherwise.
     */
    bool hasAtLeastThreeDifferentSizes() const;

    /**
     * @brief Calculate R-squared (coefficient of determination) for goodness of fit.
     *
     * R-squared ranges from 0 to 1, where 1 indicates perfect fit.
     *
     * @param[out] rSquared Calculated R-squared value.
     *
     * @return true if calculation succeeded (requires at least 2 points), false otherwise.
     */
    bool calculateRSquared(double& rSquared) const;

private:
    Mode mode_;
    std::vector<std::pair<double, double>> dataPoints;
    std::map<double, double> minTimesPerSize;  // For MIN mode: size -> min time
    double sumX, sumY, sumXY, sumX2;
    int n;
};

#endif  // LINEAR_REGRESSION_H_
