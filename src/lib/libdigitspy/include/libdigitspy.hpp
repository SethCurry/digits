#ifndef INCLUDE_LIBDIGITSPY_HPP_
#define INCLUDE_LIBDIGITSPY_HPP_
#include "Leap.h"
#include <vector>
#include <iostream>
#include <cstring>
#include <mutex>

using namespace Leap;

class LibdigitspyListener : public Listener {
  long lastEmit = 0;
  long interval = 50;
  int bufferSize = 20;
  std::vector<std::vector<float>> bufferedFrames;
  std::mutex bufferedFramesMut;

public:
    int size;
    bool full;
    void setBufferSize(int);
    void setInterval(long);
    void getFrames(float **into);
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

  private:
};

#endif
