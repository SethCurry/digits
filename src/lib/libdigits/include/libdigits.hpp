#ifndef INCLUDE_LIBDIGITS_HPP_
#define INCLUDE_LIBDIGITS_HPP_
#include "Leap.h"
#include <vector>

using namespace Leap;


const std::string fingerNames[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
const std::string boneNames[] = {"Metacarpal", "Proximal", "Middle", "Distal"};
const std::string stateNames[] = {"STATE_INVALID", "STATE_START",
                                  "STATE_UPDATE", "STATE_END"};

void logFrame(Leap::Frame frame);
void logModelFrame(Leap::Frame frame);

class Flags {
    public:
        long interval;
        int format;
};

std::vector<float> modelInputs(Hand hand);

void parse_cli_flags(int argc, const char* argv[], Flags *flags);

#define FORMAT_FULL 0
#define FORMAT_MODEL 1

class LibdigitsListener : public Listener {
    int format = 0;
    long lastEmit = 0;
    long interval = 50;

public:
    void setInterval(long);
    void setFormat(int);
    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
    virtual void onFocusGained(const Controller&);
    virtual void onFocusLost(const Controller&);
    virtual void onDeviceChange(const Controller&);
    virtual void onServiceConnect(const Controller&);
    virtual void onServiceDisconnect(const Controller&);
};

#endif
