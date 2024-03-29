# NV_system_time

Name

    NV_system_time

Name Strings

    EGL_NV_system_time

Contact

    Jason Allen, NVIDIA Corporation (jallen 'at' nvidia.com)

Status

    TBD

Version

    Version 1, July 7, 2011

Number

    EGL Extension #31

Dependencies

    Requires EGL 1.2

Overview

    This extension exposes an alternative method of querying the system time
    from the driver instead of the operating system. 

Issues

    Add 64 bit types?

      Yes, EGL doesn't support any 64 bit types so this extension adds int64
      and uint64 types.

New Types

    EGLint64NV: 64bit signed integer
    EGLuint64NV: 64bit unsigned integer

New Procedures and Functions

    EGLuint64NV eglGetSystemTimeFrequencyNV(void);
    EGLuint64NV eglGetSystemTimeNV(void);

New Tokens

    None

Description

    The command:

        EGLuint64NV eglGetSystemTimeFrequencyNV(void);

    returns the frequency of the system timer, in counts per second. The
    frequency will not change while the system is running.

    The command:

        EGLuint64NV eglGetSystemTimeNV(void);

    returns the current value of the system timer. The system time in seconds
    can be calculated by dividing the returned value by the frequency returned
    by the eglGetSystemTimeFrequencyNV command.

    Multiple calls to eglGetSystemTimeNV may return the same values, applications
    need to be careful to avoid divide by zero errors when using the interval
    calculated from successive eglGetSystemTimeNV calls.

Usage Example

    EGLuint64NV frequency = eglGetSystemTimeFrequencyNV();

    loop
    {
        EGLuint64NV start = eglGetSystemTimeNV() / frequency;

        // draw 

        EGLuint64NV end = eglGetSystemTimeNV() / frequency;

        EGLuint64NV interval = end - start;
        if (interval > 0)
            update_animation(interval);

        eglSwapBuffers(dpy, surface);
    }

Revision History

#1 (Jon Leech, 2011/07/07)
  - Add missing fields, assign extension number, and publish in the registry.

