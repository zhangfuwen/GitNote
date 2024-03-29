# ANDROID_recordable

Name

    ANDROID_recordable

Name Strings

    EGL_ANDROID_recordable

Contributors

    Jamie Gennis

Contact

    Jamie Gennis, Google Inc. (jgennis 'at' google.com)

Status

    Complete

Version

    Version 2, July 15, 2011

Number

    EGL Extension #51

Dependencies

    Requires EGL 1.0

    This extension is written against the wording of the EGL 1.4 Specification

Overview

    Android supports a number of different ANativeWindow implementations that
    can be used to create an EGLSurface.  One implementation, which records the
    rendered image as a video each time eglSwapBuffers gets called, may have
    some device-specific restrictions.  Because of this, some EGLConfigs may be
    incompatible with these ANativeWindows.  This extension introduces a new
    boolean EGLConfig attribute that indicates whether the EGLConfig supports
    rendering to an ANativeWindow that records images to a video.

New Types

    None.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <attribute> parameter of eglGetConfigAttrib and
    the <attrib_list> parameter of eglChooseConfig:

        EGL_RECORDABLE_ANDROID                      0x3142

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Section 3.4, Configuration Management, add a row to Table 3.1.
    
              Attribute             Type                 Notes
        ----------------------     -------     --------------------------
        EGL_RECORDABLE_ANDROID     boolean     whether video recording is
                                               supported

    Section 3.4, Configuration Management, add a row to Table 3.4.

              Attribute            Default     Selection  Sort   Sort
                                               Criteria   Order  Priority
        ----------------------  -------------  ---------  -----  --------
        EGL_RECORDABLE_ANDROID  EGL_DONT_CARE    Exact    None

    Section 3.4, Configuration Management, add a paragraph at the end of the
    subsection titled Other EGLConfig Attribute Descriptions.

        EGL_RECORDABLE_ANDROID is a boolean indicating whether the config may
        be used to create an EGLSurface from an ANativeWindow that is a video
        recorder as indicated by the NATIVE_WINDOW_IS_VIDEO_RECORDER query on
        the ANativeWindow.

    Section 3.4.1, Querying Configurations, change the last paragraph as follow

        EGLConfigs are not sorted with respect to the parameters
        EGL_BIND_TO_TEXTURE_RGB, EGL_BIND_TO_TEXTURE_RGBA, EGL_CONFORMANT,
        EGL_LEVEL, EGL_NATIVE_RENDERABLE, EGL_MAX_SWAP_INTERVAL,
        EGL_MIN_SWAP_INTERVAL, EGL_RENDERABLE_TYPE, EGL_SURFACE_TYPE,
        EGL_TRANSPARENT_TYPE, EGL_TRANSPARENT_RED_VALUE,
        EGL_TRANSPARENT_GREEN_VALUE, EGL_TRANSPARENT_BLUE_VALUE, and
        EGL_RECORDABLE_ANDROID.

Issues

    1. Should this functionality be exposed as a new attribute or as a bit in
    the EGL_SURFACE_TYPE bitfield?

    RESOLVED: It should be a new attribute.  It does not make sense to use up a
    bit in the limit-size bitfield for a platform-specific extension.

    2. How should the new attribute affect the sorting of EGLConfigs?

    RESOLVED: It should not affect sorting.  Some implementations may not have
    any drawback associated with using a recordable EGLConfig.  Such
    implementations should not have to double-up some of their configs to  one
    sort earlier than .  Implementations that do have drawbacks can use the
    existing caveat mechanism to report this drawback to the client.

    3. How is this extension expected to be implemented?

    RESPONSE: There are two basic approaches to implementing this extension
    that were considered during its design.  In both cases it is assumed that a
    color space conversion must be performed at some point because most video
    encoding formats use a YUV color space.  The two approaches are
    distinguished by the point at which this color space conversion is
    performed.

    One approach involves performing the color space conversion as part of the
    eglSwapBuffers call before queuing the rendered image to the ANativeWindow.
    In this case, the VisualID of the EGLConfig would correspond to a YUV
    Android HAL pixel format from which the video encoder can read.  The
    EGLConfig would likely have the EGL_SLOW_CONFIG caveat because using that
    config to render normal window contents would result in an RGB -> YUV color
    space conversion when rendering the frame as well as a YUV -> RGB
    conversion when compositing the window.

    The other approach involves performing the color space conversion in the
    video encoder.  In this case, the VisualID of the EGLConfig would
    correspond to an RGB HAL pixel format from which the video encoder can
    read.  The EGLConfig would likely not need to have any caveat set, as using
    this config for normal window rendering would not have any added cost.

Revision History

#2 (Jamie Gennis, July 15, 2011)
    - Added issue 3.

#1 (Jamie Gennis, July 8, 2011)
    - Initial draft.
