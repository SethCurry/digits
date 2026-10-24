#pragma once

#include "Leap.h"

#include <array>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

// Shared array constants. `inline` (C++17) gives them a single address across
// translation units without requiring a separate definition in a .cpp file.
inline const std::array<std::string, 5> fingerNames = {
    "Thumb", "Index", "Middle", "Ring", "Pinky"};
inline const std::array<std::string, 4> boneNames = {
    "Metacarpal", "Proximal", "Middle", "Distal"};
inline const std::array<std::string, 4> stateNames = {
    "STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END"};

inline constexpr int FORMAT_FULL = 0;
inline constexpr int FORMAT_MODEL = 1;

std::vector<float> modelInputs(Leap::Hand hand);

// Logs frames either to standard output or to a file, depending on the
// output path given at construction ("stdout" means standard output).
class ModelLogger
{
    private:
        // Owns the output file when logging to a path; unused for stdout.
        std::ofstream file;
        // Points at std::cout or at the owned `file` above.
        std::ostream* output;
    public:
        ModelLogger(std::string outPath);

        // `output` points into this object, so copying or moving a logger
        // would leave the destination's pointer dangling; it is neither
        // copyable nor movable.
        ModelLogger(const ModelLogger&) = delete;
        ModelLogger& operator=(const ModelLogger&) = delete;

        void logFrame(Leap::Frame frame) const;
        void logModelFrame(Leap::Frame frame) const;
};


class LibdigitsListener : public Leap::Listener {
    int format = FORMAT_FULL;
    long lastEmit = 0;
    long interval = 50;
    ModelLogger output;

public:
    LibdigitsListener(std::string outputPath, int format, long interval);

    void onInit(const Leap::Controller&) override;
    void onConnect(const Leap::Controller&) override;
    void onDisconnect(const Leap::Controller&) override;
    void onExit(const Leap::Controller&) override;
    void onFrame(const Leap::Controller&) override;
    void onFocusGained(const Leap::Controller&) override;
    void onFocusLost(const Leap::Controller&) override;
    void onDeviceChange(const Leap::Controller&) override;
    void onServiceConnect(const Leap::Controller&) override;
    void onServiceDisconnect(const Leap::Controller&) override;
};
