package org.apache.mahout.clustering.lda.cvb;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

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

  private static ScheduledExecutorService scheduler;

  /**
   * Constructs and starts a memory logger thread.
   *
   * @param rateInMillis how often memory info should be logged.
   */
  public static void startMemoryLogger(long rateInMillis) {
    stopMemoryLogger();
    scheduler = Executors.newScheduledThreadPool(1, new ThreadFactory() {
      private final ThreadFactory delegate = Executors.defaultThreadFactory();

      @Override
      public Thread newThread(Runnable r) {
        Thread t = delegate.newThread(r);
        t.setDaemon(true);
        return t;
      }
    });
    Runnable memoryLoogerRunnable = new Runnable() {
      public void run() {
        logMemoryStatistics();
      }
    };
    scheduler.scheduleAtFixedRate(memoryLoogerRunnable, rateInMillis, rateInMillis,
        TimeUnit.MILLISECONDS);
  }

  /**
   * Constructs and starts a memory logger thread with a logging rate of 1000 milliseconds.
   */
  public static void startMemoryLogger() {
    startMemoryLogger(1000);
  }

  /**
   * Stops the memory logger, if any, started via {@link #startMemoryLogger(long)} or
   * {@link #startMemoryLogger()}.
   */
  public static void stopMemoryLogger() {
    if (scheduler == null) {
      return;
    }
    scheduler.shutdownNow();
    scheduler = null;
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
