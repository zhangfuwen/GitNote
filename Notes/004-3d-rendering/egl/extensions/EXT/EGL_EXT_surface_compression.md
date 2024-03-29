# EXT_surface_compression

Name

    EXT_surface_compression

Name Strings

    EGL_EXT_surface_compression

Contributors

    Jan-Harald Fredriksen, Arm
    Lisa Wu, Arm
    George Liu, Arm
    Laurie Hedge, Imagination Technologies

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

IP Status

    No known IP claims.

Status

    Complete

Version

    Version 1 - November 15, 2021

Number

    EGL Extension #147

Dependencies

    These extensions are written against the wording of the EGL 1.5
    specification (August 27, 2014).

    This extension interacts with EGL_EXT_yuv_surface.

Overview

    Applications may wish to take advantage of framebuffer compression. Some
    platforms may support framebuffer compression at fixed bitrates. Such
    compression algorithms generally produce results that are visually lossless,
    but the results are typically not bit exact when compared to a non-compressed
    result.

    This extension enables applications to opt-in to fixed-rate compression
    for EGL window surfaces.

    Compression may not be supported for all framebuffer formats. It can still
    be requested for all formats and applications can query what level of compression
    was actually enabled. 

New Procedures and Functions

    EGLBoolean eglQuerySupportedCompressionRatesEXT(
           EGLDisplay dpy, EGLConfig config, const EGLAttrib *attrib_list,
           EGLint *rates, EGLint rate_size, EGLint *num_rates);

New Tokens

    New attributes accepted by the <attrib_list> argument of
    eglCreatePlatformWindowSurface and eglCreateWindowSurface:
        EGL_SURFACE_COMPRESSION_EXT                     0x34B0

    [Only if EGL_EXT_yuv_surface is supported]
        EGL_SURFACE_COMPRESSION_PLANE1_EXT              0x328E
        EGL_SURFACE_COMPRESSION_PLANE2_EXT              0x328F

    Accepted as attribute values for EGL_SURFACE_COMPRESSION_EXT by
    eglCreatePlatformWindowSurface and eglCreateWindowSurface:
        EGL_SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT     0x34B1
        EGL_SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT  0x34B2

        EGL_SURFACE_COMPRESSION_FIXED_RATE_1BPC_EXT     0x34B4
        EGL_SURFACE_COMPRESSION_FIXED_RATE_2BPC_EXT     0x34B5
        EGL_SURFACE_COMPRESSION_FIXED_RATE_3BPC_EXT     0x34B6
        EGL_SURFACE_COMPRESSION_FIXED_RATE_4BPC_EXT     0x34B7
        EGL_SURFACE_COMPRESSION_FIXED_RATE_5BPC_EXT     0x34B8
        EGL_SURFACE_COMPRESSION_FIXED_RATE_6BPC_EXT     0x34B9
        EGL_SURFACE_COMPRESSION_FIXED_RATE_7BPC_EXT     0x34BA
        EGL_SURFACE_COMPRESSION_FIXED_RATE_8BPC_EXT     0x34BB
        EGL_SURFACE_COMPRESSION_FIXED_RATE_9BPC_EXT     0x34BC
        EGL_SURFACE_COMPRESSION_FIXED_RATE_10BPC_EXT    0x34BD
        EGL_SURFACE_COMPRESSION_FIXED_RATE_11BPC_EXT    0x34BE
        EGL_SURFACE_COMPRESSION_FIXED_RATE_12BPC_EXT    0x34BF

Modifications to the EGL 1.5 Specification

    Modify section 3.5.1 "Creating On-Screen Rendering Surfaces:

    Add EGL_SURFACE_COMPRESSION_EXT to the list of attributes that can
    be specified in <attrib_list> for eglCreatePlatformWindowSurface.

    Add the following paragraph:

    EGL_SURFACE_COMPRESSION_EXT specifies the fixed-rate compression that may
    be enabled for rendering to the window.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT, then fixed-rate
    compression is disabled.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT,
    then the implementation may enable compression at a default,
    implementation-defined, rate.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_1BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 1 bit and less than 2 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_2BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 2 bits and less than 3 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_3BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 3 bits and less than 4 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_4BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 4 bits and less than 5 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_5BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 5 bits and less than 6 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_6BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 6 bits and less than 7 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_7BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 7 bits and less than 8 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_8BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 8 bits and less than 9 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_9BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 9 bits and less than 10 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_10BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 10 bits and less than 11 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_11BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 11 bits and less than 12 bits per component.
    If its value is EGL_SURFACE_COMPRESSION_FIXED_RATE_12BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 12 bits per component.

    For pixel formats with different number of bits per component, the
    specified fixed-rate compression rate applies to the component with
    the highest number of bits.

    The default value of EGL_SURFACE_COMPRESSION_EXT is
    EGL_SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT.

   [Only if EGL_EXT_yuv_surface is supported]

    If _config_ describes a surface with multiple planes (the
    value of the EGL_YUV_NUMBER_OF_PLANES_EXT attribute is larger than one),
    then the fixed-rate compression rate can be specified independently for
    the each plane.
    In this case, EGL_SURFACE_COMPRESSION_EXT specifies the fixed-rate
    compression that may be enabled for rendering to plane 0,
    EGL_SURFACE_COMPRESSION_PLANE1_EXT specifies the fixed-rate
    compression that may be enabled for rendering to plane 1, and
    EGL_SURFACE_COMPRESSION_PLANE2_EXT specifies the fixed-rate compression
    that may be enabled for rendering to plane 2.
    The supported values of EGL_SURFACE_COMPRESSION_PLANE1_EXT and
    EGL_SURFACE_COMPRESSION_PLANE2_EXT are the same as for
    EGL_SURFACE_COMPRESSION_EXT.

    If _config_ has more than one plane and the
    EGL_SURFACE_COMPRESSION_PLANE1_EXT attribute is not specified,
    then the value of EGL_SURFACE_COMPRESSION_EXT is used for all planes.
    If _config_ has more than two planes and the
    EGL_SURFACE_COMPRESSION_PLANE2_EXT attribute is not specified,
    then the value of EGL_SURFACE_COMPRESSION_PLANE1_EXT is also used
    for plane 2.

    The default value of EGL_SURFACE_COMPRESSION_PLANE1_EXT and
    EGL_SURFACE_COMPRESSION_PLANE2_EXT is EGL_SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT.

    Modify section 3.5.6 "Surface Attributes"

    Add entries to Table 3.5 "Queryable surface attributes and types":
         
        EGL_SURFACE_COMPRESSION_EXT            enum    Fixed-rate compression

   [Only if EGL_EXT_yuv_surface is supported]
        EGL_SURFACE_COMPRESSION_PLANE1_EXT     enum    Fixed-rate compression for plane 1
        EGL_SURFACE_COMPRESSION_PLANE2_EXT     enum    Fixed-rate compression for plane 2

    Add the following paragraph:

    Querying EGL_SURFACE_COMPRESSION_EXT returns the actual fixed-rate
    compression applied to a surface. For YUV surfaces, the value applied to
    the luma plane is returned. This value may be different to the one
    requested when the surface was created.
    For pbuffer and pixmap surfaces, the contents of <value> are not modified."

    [Only if EGL_EXT_yuv_surface is supported]
    Querying EGL_SURFACE_COMPRESSION_PLANE1_EXT returns the actual
    fixed-rate compression applied to plane 1 of a YUV surface.
    Querying EGL_SURFACE_COMPRESSION_PLANE2_EXT returns the actual
    fixed-rate compression applied to plane 2 of a YUV surface.
    These values may be different to the one requested when the surface was created.
    For pbuffer and pixmap surfaces, the contents of <value> are not modified."

    To get the list of all fixed-rate compression rates that are available on
    a specified display and EGLConfig, call

       EGLBoolean eglQuerySupportedCompressionRatesEXT(
           EGLDisplay dpy, EGLConfig config, const EGLAttrib *attrib_list,
           EGLint *rates, EGLint rate_size, EGLint *num_rates);

    <attrib_list> specifies a list of attributes that will be provided when a surface is created with
    this combination of display and EGLConfig. The accepted attributes are the same as for
    eglCreatePlatformWindowSurface.
    <rates> is a pointer to a buffer containing <rate_size> elements. On success, EGL_TRUE is
    returned. The number of rates is returned in <num_rates>, and elements 0 through <num_rates>-1 of
    <rates> are filled in with the available compression rates.
    No more than <rate_size> compression rates will be returned even if more are available
    on the specified display and config. However, if eglQuerySupportedCompressionRatesEXT is called with
    <rates> = NULL, then no rates are returned, but the total number of rates available will be returned
    in <num_rates>.
    The possible values returned in <rates> are the attribute values accepted for
    EGL_SURFACE_COMPRESSION_EXT by eglCreatePlatformWindowSurface and eglCreateWindowSurface, except
    EGL_SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT and EGL_SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT.

Errors

    [Only if EGL_EXT_yuv_surface is supported]
    Add to the error section of eglCreatePlatformWindowSurface:

    If the EGL_SURFACE_COMPRESSION_PLANE1_EXT attribute is specified and
    _config_ does not describe a surface with at least 2 planes (the
    EGL_YUV_NUMBER_OF_PLANES_EXT attribute is not greater than or
    equal to 2), an EGL_BAD_MATCH error is generated.

    If the EGL_SURFACE_COMPRESSION_PLANE2_EXT attribute is specified and
    _config_ does not describe a surface with at least 3 planes (the
    EGL_YUV_NUMBER_OF_PLANES_EXT attribute is not greater than or
    equal to 3), an EGL_BAD_MATCH error is generated.

    Add to the section describing eglQuerySupportedCompressionRatesEXT:

       * On failure, EGL_FALSE is returned.
       * An EGL_NOT_INITIALIZED error is generated if EGL is not initialized on <dpy>.
       * An EGL_BAD_PARAMETER error is generated if <num_rates> is NULL.

Issues

    1. Should fixed-rate compression be supported for pixmap or pbuffer
       surfaces?

       No, no use-cases have been identified for this.

    2. What is the result of querying EGL_SURFACE_COMPRESSION_EXT if
       EGL_SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT was requested?

       Resolved.
       The result will be the specific compression ratio chosen by the
       implementation, or EGL_SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT if
       no fixed-rate compression was applied.

    3. Should we expose different compressions rates per plane in this extension?

       Yes.

    4. How can an application query the set of supported compression rates?

       Resolved. Option B.

       Two options were considered.

       Option A:
       Reuse eglGetConfigAttrib(EGLDisplay dpy, EGLConfig config, EGLint attribute, EGLint *value);

       This is not ideal because:
        - the compression modes are currently tied to the surface, not the EGLConfig
        - we don't want this to affect EGLConfig selection etc.
        - this query can only return a single value, so you'd need to query each of the
          12 bit rates separately.

       Option B:
       Add a new query, specifically for the compression rates. This addresses the concerns
       with Option A, and is very similar to the mechanism used for the OpenGL ES API. Main
       downside is that it adds additional functions to the API. 

Revision History

    Version 1, 2021/11/15
      - Internal revisions
