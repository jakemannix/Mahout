package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Memory utilities.
 */
public class MemoryUtil {
  private static final Logger log = LoggerFactory.getLogger(MemoryUtil.class);

  /**
   * Logs current heap memory statistics.
   *
   * @see Runtime
   */
  public static void logMemoryStatistics() {
    Runtime runtime = Runtime.getRuntime();
    long freeBytes = runtime.freeMemory();
    long maxBytes = runtime.maxMemory();
    long totalBytes = runtime.totalMemory();
    long usedBytes = totalBytes - freeBytes;
    log.info("Memory (bytes): {} used, {} heap, {} max", new Object[] { usedBytes, totalBytes,
            maxBytes });
  }

  /**
   * Thread which periodically logs heap memory statistics via
   * {@link MemoryUtil#logMemoryStatistics()}.
   */
  public static class MemoryLoggerThread extends Thread {
    private long rateInMillis = 1000;
    private boolean paused = false;
    private boolean stopped = false;

    public MemoryLoggerThread(long rateInMillis) {
      super(MemoryLoggerThread.class.getSimpleName());
      setDaemon(true);
      this.rateInMillis = rateInMillis;
    }

    public MemoryLoggerThread() {
      this(1000);
    }

    @Override
    public synchronized void run() {
      log.info("Starting");
      while (!stopped) {
        while (!paused) {
          try {
            wait(rateInMillis);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          logMemoryStatistics();
        }
        while (paused && !stopped) {
          log.info("Paused");
          try {
            wait();
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        }
      }
      log.info("Stopping");
    }

    public long getRateInMillis() {
      return rateInMillis;
    }

    public synchronized void setRateInMillis(long rateInMillis) {
      this.rateInMillis = rateInMillis;
      notifyAll();
    }

    public boolean isPaused() {
      return paused;
    }

    public synchronized void setPaused(boolean paused) {
      this.paused = paused;
      notifyAll();
    }

    public synchronized void end() {
      stopped = paused = true;
      notifyAll();
    }
  }

  private static MemoryLoggerThread memoryLoggerThread;

  /**
   * Constructs and starts a {@link MemoryLoggerThread}.
   *
   * @param rateInMillis how often memory info should be logged.
   */
  public static synchronized void startMemoryLogger(long rateInMillis) {
    Preconditions.checkState(memoryLoggerThread == null, "Memory logger already started");
    memoryLoggerThread = new MemoryLoggerThread(rateInMillis);
    memoryLoggerThread.start();
  }

  /**
   * Constructs and starts a {@link MemoryLoggerThread} with a logging rate of 1000 milliseconds.
   */
  public static synchronized void startMemoryLogger() {
    startMemoryLogger(1000);
  }

  /**
   * Stops the {@link MemoryLoggerThread}, if any, started via {@link #startMemoryLogger(long)} or
   * {@link #startMemoryLogger()}.
   */
  public static synchronized void stopMemoryLogger() {
    if (memoryLoggerThread == null) {
      return;
    }
    memoryLoggerThread.end();
    memoryLoggerThread = null;
  }

  /**
   * Tests {@link MemoryLoggerThread}.
   *
   * @param args
   * @throws InterruptedException
   */
  public static void main(String[] args) throws InterruptedException {
    startMemoryLogger();
    Thread.sleep(10000);
  }
}
