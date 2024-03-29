# ANDROID_get_frame_timestamps

Name

    ANDROID_get_frame_timestamps

Name Strings

    EGL_ANDROID_get_frame_timestamps

Contributors

    Brian Anderson
    Dan Stoza
    Pablo Ceballos
    Jesse Hall
    Fabien Sanglard

Contact

    Brian Anderson, Google Inc. (brianderson 'at' google.com)
    Dan Stoza, Google Inc. (stoza 'at' google.com)
    Pablo Ceballos, Google Inc. (pceballos 'at' google.com)
    Jesse Hall, Google Inc. (jessehall 'at' google.com)
    Fabien Sanglard, Google Inc. (sanglardf 'at' google.com)

Status

    Draft

Version

    Version 8, April 11, 2017

Number

    EGL Extension #122

Dependencies

    Requires EGL 1.2

    This extension is written against the wording of the EGL 1.5 Specification

Overview

    This extension allows querying various timestamps related to the composition
    and display of window surfaces.

    Some examples of how this might be used:
        - The display present time can be used to calculate end-to-end latency
          of the entire graphics pipeline.
        - The queue time and rendering complete time can be used to determine
          how long the application's rendering took to complete. Likewise, the
          composition start time and finish time can be used to determine how
          long the compositor's rendering work took. In combination these can be
          used to help determine if the system is GPU or CPU bound.

New Types

    /*
     * EGLnsecsANDROID is a signed integer type for representing a time in
     * nanoseconds.
     */
    #include <khrplatform.h>
    typedef khronos_stime_nanoseconds_t EGLnsecsANDROID;

New Procedures and Functions

    EGLBoolean eglGetNextFrameIdANDROID(EGLDisplay dpy, EGLSurface surface,
            EGLuint64KHR *frameId);

    EGLBoolean eglGetCompositorTimingANDROID(EGLDisplay dpy,
            EGLSurface surface, EGLint numTimestamps,
            const EGLint *names, EGLnsecsANDROID *values);

    EGLBoolean eglGetFrameTimestampsANDROID(EGLDisplay dpy, EGLSurface surface,
            EGLuint64KHR frameId, EGLint numTimestamps,
            const EGLint *timestamps, EGLnsecsANDROID *values);

    EGLBoolean eglGetFrameTimestampSupportedANDROID(EGLDisplay dpy,
            EGLSurface surface, EGLint timestamp);

New Tokens

    EGL_TIMESTAMPS_ANDROID 0x3430
    EGL_COMPOSITE_DEADLINE_ANDROID 0x3431
    EGL_COMPOSITE_INTERVAL_ANDROID 0x3432
    EGL_COMPOSITE_TO_PRESENT_LATENCY_ANDROID 0x3433
    EGL_REQUESTED_PRESENT_TIME_ANDROID 0x3434
    EGL_RENDERING_COMPLETE_TIME_ANDROID 0x3435
    EGL_COMPOSITION_LATCH_TIME_ANDROID 0x3436
    EGL_FIRST_COMPOSITION_START_TIME_ANDROID 0x3437
    EGL_LAST_COMPOSITION_START_TIME_ANDROID 0x3438
    EGL_FIRST_COMPOSITION_GPU_FINISHED_TIME_ANDROID 0x3439
    EGL_DISPLAY_PRESENT_TIME_ANDROID 0x343A
    EGL_DEQUEUE_READY_TIME_ANDROID 0x343B
    EGL_READS_DONE_TIME_ANDROID 0x343C
    EGL_TIMESTAMP_PENDING_ANDROID -2
    EGL_TIMESTAMP_INVALID_ANDROID -1

Add to the list of supported tokens for eglSurfaceAttrib in section 3.5.6
"Surface Attributes", page 43:

    If attribute is EGL_TIMESTAMPS_ANDROID, then values specifies whether to
    enable/disable timestamp collection for this surface. A value of EGL_TRUE
    enables timestamp collection, while a value of EGL_FALSE disables it. The
    initial value is false. If surface is not a window surface this has no
    effect.
Changes to Chapter 3 of the EGL 1.5 Specification (EGL Functions and Errors)

    Add a new subsection under Section 3,

    "3.13 Composition and Display Timestamps

    The function

        EGLBoolean eglGetNextFrameIdANDROID(EGLDisplay dpy, EGLSurface surface,
            EGLuint64KHR *frameId);

    Returns an identifier for the next frame to be swapped. The identifier can
    be used to correlate a particular eglSwapBuffers with its timestamps in
    eglGetFrameTimestampsANDROID. If any error is generated, the function will
    return EGL_FALSE.

    The function

        EGLBoolean eglGetCompositorTimingANDROID(EGLDisplay dpy,
                EGLSurface surface, EGLint numTimestamps,
                const EGLint *names, EGLnsecsANDROID *values);

    allows querying anticipated timestamps and durations related to the
    composition and display of a window surface. The values are not associated
    with a particular frame and can be retrieved before the first swap.

    The eglGetCompositorTimingANDROID function takes an array of names to
    query and returns their values in the corresponding indices of the values
    array. The possible names that can be queried are:
        - EGL_COMPOSITE_DEADLINE_ANDROID - The timestamp of the next time the
          compositor will begin composition. This is effectively the deadline
          for when the compositor must receive a newly queued frame.
        - EGL_COMPOSITE_INTERVAL_ANDROID - The time delta between subsequent
          composition events.
        - EGL_COMPOSITE_TO_PRESENT_LATENCY_ANDROID - The time delta between
          the start of composition and the expected present time of that
          composition. This can be used to estimate the latency of the
          actual present time.

    The function

        EGLBoolean eglGetFrameTimestampsANDROID(EGLDisplay dpy,
            EGLSurface surface, EGLuint64KHR frameId, EGLint numTimestamps,
            const EGLint *timestamps, EGLnsecsANDROID *values);

    allows querying various timestamps related to the composition and display
    of specific frames of a window surface.

    The frameId indicates which frame to query. The implementation maintains a
    limited history of timestamp data. If a query is made for a frame whose
    timestamp history no longer exists then EGL_BAD_ACCESS is generated. If
    timestamp collection has not been enabled for the surface then
    EGL_BAD_SURFACE is generated.  Timestamps for events that might still occur
    will have the value EGL_TIMESTAMP_PENDING_ANDROID. Timestamps for events
    that did not occur will have the value EGL_TIMESTAMP_INVALID_ANDROID.
    Otherwise, the timestamp will be valid and indicate the event has occured.
    Timestamp queries that are not supported will generate an EGL_BAD_PARAMETER
    error. If any error is generated the function will return EGL_FALSE.

    The application can poll for the timestamp of particular events by calling
    eglGetFrameTimestamps over and over without needing to call any other EGL
    function between calls. This is true even for the most recently swapped
    frame. eglGetFrameTimestamps is thread safe and can be called from a
    different thread than the swapping thread.

    The eglGetFrameTimestampsANDROID function takes an array of timestamps to
    query and returns timestamps in the corresponding indices of the values
    array. The possible timestamps that can be queried are:
        - EGL_REQUESTED_PRESENT_TIME_ANDROID - The time the application
          requested this frame be presented. See EGL_ANDROID_presentation_time.
          If the application does not request a presentation time explicitly,
          this will correspond to buffer's queue time.
        - EGL_RENDERING_COMPLETE_TIME_ANDROID - The time when all of the
          application's rendering to the surface was completed.
        - EGL_COMPOSITION_LATCH_TIME_ANDROID - The time when the compositor
          selected this frame as the one to use for the next composition. The
          latch is the earliest indication that the frame was submitted in time
          to be composited.
        - EGL_FIRST_COMPOSITION_START_TIME_ANDROID - The first time at which
          the compositor began preparing composition for this frame.
        - EGL_LAST_COMPOSITION_START_TIME_ANDROID - The last time at which the
          compositor began preparing composition for this frame. If this frame
          is composited only once, it will have the same value as
          EGL_FIRST_COMPOSITION_START_TIME_ANDROID. If the value is not equal,
          that indicates the subsequent frame was not submitted in time to be
          latched by the compositor. Note: The value may not be updated for
          every display refresh if the compositor becomes idle.
        - EGL_FIRST_COMPOSITION_GPU_FINISHED_TIME_ANDROID - The time at which
          the compositor's rendering work for this frame finished. This will be
          zero if composition was handled by the display and the compositor
          didn't do any rendering.
        - EGL_DISPLAY_PRESENT_TIME_ANDROID - The time at which this frame
          started to scan out to the physical display.
        - EGL_DEQUEUE_READY_TIME_ANDROID - The time when the buffer became
          available for reuse as a buffer the client can target without
          blocking. This is generally the point when all read commands of the
          buffer have been submitted, but not necessarily completed.
        - EGL_READS_DONE_TIME_ANDROID - The time at which all reads for the
          purpose of display/composition were completed for this frame.

    Not all implementations may support all of the above timestamp queries. The
    functions

        EGLBoolean eglGetCompositorTimingSupportedANDROID(EGLDisplay dpy,
            EGLSurface surface, EGLint name);

    and

        EGLBoolean eglGetFrameTimestampSupportedANDROID(EGLDisplay dpy,
            EGLSurface surface, EGLint timestamp);

    allows querying which values are supported by the implementations of
    eglGetCompositorTimingANDROID and eglGetFrameTimestampSupportedANDROID
    respectively."

Issues

    None

Revision History

#9 (Chris Forbes, June 11, 2019)
    - Fix eglGetFrameTimestampSupportedANDROID function name in extension
      spec to match reality

#8 (Brian Anderson, April 11, 2017)
    - Use reserved enumerant values.

#7 (Brian Anderson, March 21, 2017)
    - Differentiate between pending events and events that did not occur.

#6 (Brian Anderson, March 16, 2017)
    - Remove DISPLAY_RETIRE_TIME_ANDROID.

#5 (Brian Anderson, January 13, 2017)
    - Add eglGetCompositorTimingANDROID.

#4 (Brian Anderson, January 10, 2017)
    - Use an absolute frameId rather than a relative framesAgo.

#3 (Brian Anderson, November 30, 2016)
    - Add EGL_COMPOSITION_LATCH_TIME_ANDROID,
      EGL_LAST_COMPOSITION_START_TIME_ANDROID, and
      EGL_DEQUEUE_READY_TIME_ANDROID.

#2 (Brian Anderson, July 22, 2016)
    - Replace EGL_QUEUE_TIME_ANDROID with EGL_REQUESTED_PRESENT_TIME_ANDROID.
    - Add DISPLAY_PRESENT_TIME_ANDROID.

#1 (Pablo Ceballos, May 31, 2016)
    - Initial draft.

