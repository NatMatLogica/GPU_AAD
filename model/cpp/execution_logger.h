#pragma once
#include <string>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace simm {

// CSV column headers for SIMM execution log
inline const char* SIMM_LOG_HEADERS =
    "timestamp,model_name,model_version,mode,"
    "num_trades,num_threads,"
    "portfolio_npv,simm_total,"
    "ir_delta_margin,eq_delta_margin,eq_vega_margin,"
    "fx_delta_margin,fx_vega_margin,inflation_margin,"
    "eval_time_sec,recording_time_sec,kernel_memory_mb,"
    "language,uses_aadc,vector_size,status\n";

class SIMMExecutionLogger {
public:
    explicit SIMMExecutionLogger(const std::string& log_file = "data/execution_log.csv")
        : log_file_(log_file) {
        ensure_log_file_exists();
    }

    void log(const std::string& model_name,
             const std::string& model_version,
             const std::string& mode,
             int num_trades,
             int num_threads,
             double portfolio_npv,
             double simm_total,
             double ir_delta_margin,
             double eq_delta_margin,
             double eq_vega_margin,
             double fx_delta_margin,
             double fx_vega_margin,
             double inflation_margin,
             double eval_time_sec,
             double recording_time_sec,
             double kernel_memory_mb,
             const std::string& language,
             bool uses_aadc,
             int vector_size) {
        std::ofstream f(log_file_, std::ios::app);
        if (!f.is_open()) return;

        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        std::ostringstream timestamp;
        timestamp << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");

        f << timestamp.str() << ","
          << model_name << ","
          << model_version << ","
          << mode << ","
          << num_trades << ","
          << num_threads << ","
          << std::fixed << std::setprecision(2)
          << portfolio_npv << ","
          << simm_total << ","
          << ir_delta_margin << ","
          << eq_delta_margin << ","
          << eq_vega_margin << ","
          << fx_delta_margin << ","
          << fx_vega_margin << ","
          << inflation_margin << ","
          << std::setprecision(6) << eval_time_sec << ","
          << recording_time_sec << ","
          << std::setprecision(2) << kernel_memory_mb << ","
          << language << ","
          << (uses_aadc ? "yes" : "no") << ","
          << vector_size << ","
          << "success\n";
        f.close();
    }

private:
    std::string log_file_;

    void ensure_log_file_exists() {
        struct stat buffer;
        if (stat(log_file_.c_str(), &buffer) != 0) {
            // Create parent directory if needed
            std::string dir = log_file_.substr(0, log_file_.find_last_of('/'));
            if (!dir.empty()) {
                mkdir(dir.c_str(), 0755);
            }
            std::ofstream f(log_file_);
            if (f.is_open()) {
                f << SIMM_LOG_HEADERS;
                f.close();
            }
        }
    }
};

}  // namespace simm
