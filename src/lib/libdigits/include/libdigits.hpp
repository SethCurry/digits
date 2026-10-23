#pragma once

#include "Leap.h"

#include <array>
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

// Command-line flags shared between the CLI front-end and the listener.
struct Flags {
    long interval = 50;
    int format = FORMAT_FULL;
};

void logFrame(Leap::Frame frame);
void logModelFrame(Leap::Frame frame);

std::vector<float> modelInputs(Leap::Hand hand);

// Parses recognized flags (--interval, --format) from argv into `flags`.
void parse_cli_flags(int argc, const char* argv[], Flags& flags);

class LibdigitsListener : public Leap::Listener {
    int format = FORMAT_FULL;
    long lastEmit = 0;
    long interval = 50;

public:
    void setInterval(long newInterval);
    void setFormat(int newFormat);

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