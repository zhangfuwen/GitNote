# NV_stream_consumer_gltexture_yuv

Name

    NV_stream_consumer_gltexture_yuv

Name Strings

    EGL_NV_stream_consumer_gltexture_yuv

Contributors

    James Jones
    Daniel Kartch
    Nikhil Mahale
    Daniel Koch
    Jeff Gilbert

Contacts

    James Jones, NVIDIA (jajones 'at' nvidia 'dot' com)

Status

    Complete

Version

    Version 4 - November 14, 2017

Number

    EGL Extension #94

Extension Type

    EGL display extension

Dependencies

    Requires EGL_KHR_stream_consumer_gltexture
    References EGL_EXT_yuv_surface

Interactions with EGL_EXT_yuv_surface

    This extension makes use of several tokens defined in the
    EGL_EXT_yuv_surface extension spec. However support for this
    EGLStream extension does not require the EGLSurface extension, or
    vice versa. Only the tokens are shared.

Overview

    The EGL_KHR_stream_consumer_gltexture extension allows EGLStream
    frames to be latched to a GL texture for use in rendering. These
    frames are assumed to be stored in RGB format and accessed as such
    by shader programs. If the producer uses a different color space,
    the stream implementation must perform an implicit conversion.

    In cases where the producer operates in a native YUV color space, it
    may be desirable for shaders to directly access the YUV components,
    without conversion. This extension adds a new variant of the
    function to bind GL textures as stream consumers which allows
    attributes to specify the color space.

New Types

    None

New Functions

    EGLBoolean eglStreamConsumerGLTextureExternalAttribsNV(
                    EGLDisplay       dpy,
                    EGLStreamKHR     stream,
                    const EGLAttrib *attrib_list)

New Tokens

    Accepted as attribute name in <attrib_list> by
    eglStreamConsumerGLTextureExternalAttribsNV:

        EGL_YUV_PLANE0_TEXTURE_UNIT_NV                  0x332C
        EGL_YUV_PLANE1_TEXTURE_UNIT_NV                  0x332D
        EGL_YUV_PLANE2_TEXTURE_UNIT_NV                  0x332E

Reused Tokens From EGL_EXT_yuv_surface

    Accepted as attribute name in <attrib_list> by
    eglStreamConsumerGLTextureExternalAttribsNV:

        EGL_YUV_NUMBER_OF_PLANES_EXT                    0x3311

    Accepted as value for EGL_COLOR_BUFFER_TYPE attribute in
    <attrib_list> by eglStreamConsumerGLTextureExternalAttribsNV:

        EGL_YUV_BUFFER_EXT                              0x3300

Replace entire description of eglStreamConsumerGLTextureExternalKHR in
section "3.10.2.1 GL Texture External consumer" of
EGL_KHR_stream_consumer_gltexture extension.

    Call

        EGLBoolean eglStreamConsumerGLTextureExternalAttribsNV(
                    EGLDisplay       dpy,
                    EGLStreamKHR     stream,
                    const EGLAttrib *attrib_list)

    to connect one or more texture objects in the OpenGL or OpenGL ES
    context current to the calling thread as the consumer(s) of
    <stream>. The identity and format of the texture objects used are
    determined by <attrib_list> and the current context state.

    <attrib_list> must either be NULL or point to an array of name/value
    pairs terminated by EGL_NONE. Valid attribute names are
    EGL_COLOR_BUFFER_TYPE, EGL_YUV_NUMBER_OF_PLANES_EXT, and
    EGL_YUV_PLANE<n>_TEXTURE_UNIT_NV.

    If the value of EGL_COLOR_BUFFER_TYPE is EGL_RGB_BUFFER (the
    default), then the stream will be connected to a single texture
    whose contents are available to shaders as RGB values. If the value
    of EGL_COLOR_BUFFER_TYPE is EGL_YUV_BUFFER_EXT the stream will be
    connected to some number of planar textures, determined by the value
    of EGL_YUV_NUMBER_OF_PLANES_EXT, whose contents are available to
    shaders as YUV values. The mapping between YUV values and texture
    contents is described in table 3.10.2.1.

    If EGL_COLOR_BUFFER_TYPE is EGL_YUV_BUFFER_EXT, the default value of
    EGL_YUV_NUMBER_OF_PLANES_EXT is 2. Otherwise it is 0.

                    PLANE0            PLANE1            PLANE2
    # Planes    Values  Fields    Values  Fields    Values  Fields
    --------------------------------------------------------------
        1         YUV     XYZ         unused            unused
        2         Y       X         UV      XY          unused
        3         Y       X         U       X         V       X

                Table 3.10.2.1 YUV Planar Texture Mappings

    If EGL_COLOR_BUFFER_TYPE is EGL_RGB_BUFFER, the stream is connected
    to the texture object currently bound to the active texture unit's
    GL_TEXTURE_EXTERNAL_OES texture target in the current context.

    If EGL_COLOR_BUFFER_TYPE is EGL_YUV_BUFFER_EXT, attribute values
    must be specified for EGL_YUV_PLANE<n>_TEXTURE_UNIT_NV for all <n>
    less than the number of planes. The value of each attribute must
    either be a valid texture unit index or EGL_NONE. No two of these
    attributes may specify the same valid texture unit index or
    reference separate texture units bound to the same texture object.
    Plane <n> of the stream contents will be connected to the texture
    object currently bound to the indexed texture unit's
    GL_TEXTURE_EXTERNAL_OES texture target in the current context, or
    will be left unused if the index is EGL_NONE.

    Once connected, the stream will remain associated with the initial
    texture object(s) even if the texture units are bound to new
    textures.

    (Note: Before this can succeed a GL_TEXTURE_EXTERNAL_OES texture
    must be bound to the appropriate texture units of the GL context
    current to the calling thread.  To create a GL_TEXTURE_EXTERNAL_OES
    texture and bind it to the current context, call glBindTexture()
    with <target> set to GL_TEXTURE_EXTERNAL_OES and <texture> set to
    the name of the GL_TEXTURE_EXTERNAL_OES (which may or may not have
    previously been created).  This is described in the
    GL_NV_EGL_stream_consumer_external extension.)

    On failure EGL_FALSE is returned and an error is generated.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          EGLDisplay.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStreamKHR created for <dpy>.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_CREATED_KHR.

        - EGL_BAD_ATTRIBUTE is generated if any attribute name in
          <attrib_list> is not a valid attribute.

        - EGL_BAD_PARAMETER is generated if the value of
          EGL_COLOR_BUFFER_TYPE is not EGL_RGB_BUFFER or
          EGL_YUV_BUFFER_EXT.

        - EGL_BAD_MATCH is generated if EGL_COLOR_BUFFER_TYPE is
          EGL_RGB_BUFFER and EGL_YUV_NUMBER_OF_PLANES_EXT is not 0, or
          if EGL_COLOR_BUFFER_TYPE is EGL_YUV_BUFFER_EXT and
          EGL_YUV_NUMBER_OF_PLANES_EXT is not 1, 2, or 3.

        - EGL_BAD_MATCH is generated if any
          EGL_YUV_PLANE<n>_TEXTURE_UNIT_NV is not specified for any <n>
          less than EGL_YUV_NUMBER_OF_PLANES_EXT, or if it is specified
          for any <n> greater than or equal to
          EGL_YUV_NUMBER_OF_PLANES_EXT.

        - EGL_BAD_ACCESS is generated if any
          EGL_YUV_PLANE<n>_TEXTURE_UNIT_NV is set to anything other than
          a valid texture unit index or EGL_NONE.

        - EGL_BAD_ACCESS is generated if there is no GL context
          current to the calling thread.

        - EGL_BAD_ACCESS is generated unless nonzero texture object
          names are bound the GL_TEXTURE_EXTERNAL_OES texture target
          of each of the appropriate texture units of the GL context
          current to the calling thread.

        - EGL_BAD_ACCESS is generated if more than one planar surface
          would be bound to the same texture object.

        - EGL_BAD_ACCESS is generated if the implementation cannot
          support the requested planar arrangement.

    On success the texture(s) are connected to the <stream>, <stream>
    is placed in the EGL_STREAM_STATE_CONNECTING_KHR state, and EGL_TRUE
    is returned.

    When a producer is later connected, if it cannot support the planar
    arrangement of the GL texture connection, it will fail with an
    EGL_BAD_ACCESS error.

    If any texture is later deleted, connected to a different
    EGLStream, or connected to an EGLImage, then <stream> will be
    placed into the EGL_STREAM_STATE_DISCONNECTED_KHR state.

    If the <stream> is later destroyed then the textures will be
    "incomplete" until they are connected to a new EGLStream, connected
    to a new EGLImage, or deleted.

    The function

        EGLBoolean eglStreamConsumerGLTextureExternalKHR(
                    EGLDisplay    dpy,
                    EGLStreamKHR  stream)

    is equivalent to eglStreamConsumerGLTextureExternalAttribsNV with
    <attrib_list> list set to NULL.

In the remainder of section "3.10.2.1 GL Texture External consumer",
replace all singular references to "texture" with "textures" and make
appropriate grammatical modifications.

Issues

    1.  This competes with GL_EXT_yuv_target as a means for specifying
        how YUV values can be directly accessed by a texture shader
        without conversion to RGB. However, that extension also requires
        a means to render to YUV surfaces in addition to using them as
        textures. Should we go with the approach used here or create a
        GL extension which defines a subset GL_EXT_yuv_target?

        RESOLVED: The extension as is serves immediate needs. Conflicts
        and overlap with other extensions will be addressed if and when
        there is a need to promote to EXT.

    2.  This also contradicts how previous extensions for EXTERNAL GL
        textures bind multiplanar surfaces, using separate texture
        objects rather than a single virtual texture object which
        requires multiple texture units. This allows the application
        greater control of the planar arrangement, and the ability to
        leave planes unbound, which may reduce overhead for the
        producer. But it makes applications less portabile if the
        desired arrangement isn't supported.

        RESOLVED: The extension as is serves immediate needs. Conflicts
        and overlap with other extensions will be addressed if and when
        there is a need to promote to EXT.

Revision History

    #4  (November 14, 2017) Mozilla Corporation
        - Const-qualify attrib_list.

    #3  (August 19, 2015) NVIDIA Corporation
        - Added enum values.
        - Cleaned up and added contact info for publication.

    #2  (May 6, 2015) NVIDIA Corporation
        - Consolidated error codes to make GL interaction simpler.

    #1  (April 15, 2015) NVIDIA Corporation
        - Initial draft
