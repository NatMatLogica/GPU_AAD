// Data Import: Load pre-computed sensitivity matrix, factor metadata, and
// allocation from CSV files exported by benchmark_trading_workflow.py.
// This enables apples-to-apples comparison with the Python/GPU backends.
// Version: 1.0.0
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <chrono>

#include "sensitivity_matrix.h"
#include "factor_metadata.h"
#include "allocation_optimizer.h"

namespace simm {

// ============================================================================
// CSV Parsing Helpers
// ============================================================================

inline std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    // Handle trailing comma (empty last field)
    if (!line.empty() && line.back() == ',') {
        fields.push_back("");
    }
    return fields;
}

inline std::string trimStr(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ============================================================================
// String -> Enum Conversions
// ============================================================================

inline RiskClass parseRiskClass(const std::string& s) {
    if (s == "Rates")       return RiskClass::Rates;
    if (s == "CreditQ")     return RiskClass::CreditQ;
    if (s == "CreditNonQ")  return RiskClass::CreditNonQ;
    if (s == "Equity")      return RiskClass::Equity;
    if (s == "Commodity")   return RiskClass::Commodity;
    if (s == "FX")          return RiskClass::FX;
    throw std::runtime_error("Unknown risk class: " + s);
}

inline RiskMeasure parseRiskMeasure(const std::string& s) {
    if (s == "Delta") return RiskMeasure::Delta;
    if (s == "Vega")  return RiskMeasure::Vega;
    throw std::runtime_error("Unknown risk measure: " + s);
}

// ============================================================================
// Load Trade IDs (one per line, no header)
// ============================================================================

inline std::vector<std::string> loadTradeIds(const std::string& dir) {
    std::string path = dir + "/trade_ids.csv";
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::vector<std::string> ids;
    std::string line;
    while (std::getline(f, line)) {
        std::string trimmed = trimStr(line);
        if (!trimmed.empty())
            ids.push_back(trimmed);
    }
    return ids;
}

// ============================================================================
// Load Risk Factors (header + K rows: risk_type,qualifier,bucket,label1)
// ============================================================================

inline std::vector<RiskFactorKey> loadRiskFactors(const std::string& dir) {
    std::string path = dir + "/risk_factors.csv";
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::vector<RiskFactorKey> rf;
    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (trimStr(line).empty()) continue;
        auto fields = splitCSV(line);
        std::string risk_type = trimStr(fields[0]);
        std::string qualifier = fields.size() > 1 ? trimStr(fields[1]) : "";
        int bucket = 0;
        if (fields.size() > 2 && !trimStr(fields[2]).empty())
            bucket = std::stoi(trimStr(fields[2]));
        std::string label1 = fields.size() > 3 ? trimStr(fields[3]) : "";
        rf.emplace_back(risk_type, qualifier, bucket, label1);
    }
    return rf;
}

// ============================================================================
// Load Sensitivity Matrix (line 1: T,K then T rows of K doubles)
// ============================================================================

inline SensitivityMatrix loadSensitivityMatrix(const std::string& dir) {
    auto trade_ids = loadTradeIds(dir);
    auto risk_factors = loadRiskFactors(dir);

    std::string path = dir + "/sensitivity_matrix.csv";
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::string header;
    std::getline(f, header);
    auto dims = splitCSV(header);
    int T = std::stoi(trimStr(dims[0]));
    int K = std::stoi(trimStr(dims[1]));

    if (T != static_cast<int>(trade_ids.size()))
        throw std::runtime_error("T mismatch: matrix says " + std::to_string(T)
            + " but trade_ids has " + std::to_string(trade_ids.size()));
    if (K != static_cast<int>(risk_factors.size()))
        throw std::runtime_error("K mismatch: matrix says " + std::to_string(K)
            + " but risk_factors has " + std::to_string(risk_factors.size()));

    SensitivityMatrix sm;
    sm.T = T;
    sm.K = K;
    sm.trade_ids = std::move(trade_ids);
    sm.risk_factors = std::move(risk_factors);
    sm.data.resize(T * K);

    std::string line;
    for (int t = 0; t < T; ++t) {
        if (!std::getline(f, line))
            throw std::runtime_error("Premature EOF at row " + std::to_string(t));
        auto fields = splitCSV(line);
        if (static_cast<int>(fields.size()) < K)
            throw std::runtime_error("Row " + std::to_string(t) + " has "
                + std::to_string(fields.size()) + " fields, expected " + std::to_string(K));
        for (int k = 0; k < K; ++k) {
            sm.data[t * K + k] = std::stod(fields[k]);
        }
    }
    return sm;
}

// ============================================================================
// Load Factor Metadata (header + K rows, 10 fields each)
// ============================================================================

inline std::vector<FactorMeta> loadFactorMetadata(const std::string& dir) {
    std::string path = dir + "/factor_metadata.csv";
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::vector<FactorMeta> metadata;
    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (trimStr(line).empty()) continue;
        auto fields = splitCSV(line);
        if (fields.size() < 10)
            throw std::runtime_error("factor_metadata row has " +
                std::to_string(fields.size()) + " fields, expected 10: " + line);

        FactorMeta fm;
        fm.risk_class   = parseRiskClass(trimStr(fields[0]));
        fm.risk_measure = parseRiskMeasure(trimStr(fields[1]));
        fm.risk_type    = trimStr(fields[2]);
        fm.qualifier    = trimStr(fields[3]);
        fm.bucket       = std::stoi(trimStr(fields[4]));
        fm.label1       = trimStr(fields[5]);
        fm.tenor_idx    = std::stoi(trimStr(fields[6]));
        fm.weight       = std::stod(trimStr(fields[7]));
        fm.cr           = std::stod(trimStr(fields[8]));
        fm.bucket_key   = trimStr(fields[9]);
        metadata.push_back(fm);
    }
    return metadata;
}

// ============================================================================
// Load Allocation Matrix (line 1: T,P then T rows of P doubles)
// ============================================================================

inline AllocationMatrix loadAllocation(const std::string& dir) {
    std::string path = dir + "/allocation.csv";
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::string header;
    std::getline(f, header);
    auto dims = splitCSV(header);
    int T = std::stoi(trimStr(dims[0]));
    int P = std::stoi(trimStr(dims[1]));

    AllocationMatrix alloc(T, P);
    std::string line;
    for (int t = 0; t < T; ++t) {
        if (!std::getline(f, line))
            throw std::runtime_error("Premature EOF at row " + std::to_string(t));
        auto fields = splitCSV(line);
        if (static_cast<int>(fields.size()) < P)
            throw std::runtime_error("Allocation row " + std::to_string(t) + " has "
                + std::to_string(fields.size()) + " fields, expected " + std::to_string(P));
        for (int p = 0; p < P; ++p) {
            alloc(t, p) = std::stod(fields[p]);
        }
    }
    return alloc;
}

} // namespace simm
