# QCOM_extended_get

Name

    QCOM_extended_get

Name Strings

    GL_QCOM_extended_get

Contributors

    Jukka Liimatta
    James Ritts

Contact

    Jukka Liimatta (jukka.liimatta 'at' qualcomm.com)

Notice

    Copyright Qualcomm 2009.

IP Status

    Qualcomm Proprietary.

Status

    Complete.

Version

    Last Modified Date: May 14, 2009
    Revision: #1

Number

    OpenGL ES Extension #62

Dependencies

    OpenGL ES 1.0 or higher is required.

Overview

    This extension enables instrumenting the driver for debugging of OpenGL ES 
    applications.

New Procedures and Functions

    void ExtGetTexturesQCOM(uint* textures,
                            int maxTextures,
                            int* numTextures);

    void ExtGetBuffersQCOM(uint* buffers,
                           int maxBuffers,
                           int* numBuffers);

    void ExtGetRenderbuffersQCOM(uint* renderbuffers,
                                 int maxRenderbuffers,
                                 int* numRenderbuffers);

    void ExtGetFramebuffersQCOM(uint* framebuffers,
                                int maxFramebuffers,
                                int* numFramebuffers);

    void ExtGetTexLevelParameterivQCOM(uint texture, enum face, int level,
                                       enum pname, int* params);

    void ExtTexObjectStateOverrideiQCOM(enum target, enum pname, int param);

    void ExtGetTexSubImageQCOM(enum target, int level,
                               int xoffset, int yoffset, int zoffset,
                               sizei width, sizei height, sizei depth,
                               enum format, enum type, void *texels);

    void ExtGetBufferPointervQCOM(enum target, void **params)

New Tokens

    Accepted by the <pname> parameter of ExtGetTexLevelParameterivQCOM

        TEXTURE_WIDTH_QCOM                         0x8BD2
        TEXTURE_HEIGHT_QCOM                        0x8BD3
        TEXTURE_DEPTH_QCOM                         0x8BD4
        TEXTURE_INTERNAL_FORMAT_QCOM               0x8BD5
        TEXTURE_FORMAT_QCOM                        0x8BD6
        TEXTURE_TYPE_QCOM                          0x8BD7
        TEXTURE_IMAGE_VALID_QCOM                   0x8BD8
        TEXTURE_NUM_LEVELS_QCOM                    0x8BD9
        TEXTURE_TARGET_QCOM                        0x8BDA
        TEXTURE_OBJECT_VALID_QCOM                  0x8BDB

    Accepted by the <target> parameter of ExtTexObjectStateOverrideiQCOM

        TEXTURE_2D
        TEXTURE_CUBE_MAP
        TEXTURE_3D_OES
        TEXTURE_EXTERNAL_OES

    Accepted by the <pname> parameter of ExtTexObjectStateOverrideiQCOM

        STATE_RESTORE                              0x8BDC
        TEXTURE_MIN_FILTER
        TEXTURE_MAG_FILTER
        TEXTURE_WIDTH_QCOM
        TEXTURE_HEIGHT_QCOM
        TEXTURE_DEPTH_QCOM

    Accepted by the <target> parameter of ExtGetTexSubImageQCOM

        TEXTURE_2D
        TEXTURE_3D
        TEXTURE_CUBE_MAP_POSITIVE_X
        TEXTURE_CUBE_MAP_NEGATIVE_X
        TEXTURE_CUBE_MAP_POSITIVE_Y
        TEXTURE_CUBE_MAP_NEGATIVE_Y
        TEXTURE_CUBE_MAP_POSITIVE_Z
        TEXTURE_CUBE_MAP_NEGATIVE_Z

    Accepted by the <format> parameter of ExtGetTexSubImageQCOM

        ALPHA
        LUMINANCE
        LUMINANCE_ALPHA
        RGB
        RGBA
        ATC_RGB_AMD
        ATC_RGBA_EXPLICIT_ALPHA_AMD
        ATC_RGBA_INTERPOLATED_ALPHA_AMD

    Accepted by the <type> parameter of ExtGetTexSubImageQCOM

        UNSIGNED_BYTE
        UNSIGNED_SHORT_5_6_5
        UNSIGNED_SHORT_5_5_5_1
        UNSIGNED_SHORT_4_4_4_4
        HALF_FLOAT_OES
        FLOAT

    Accepted by the <target> parameter of ExtGetBufferPointervQCOM

        ARRAY_BUFFER
        ELEMENT_ARRAY_BUFFER

Additions to OpenGL ES 1.1 Specification

    The command

        void ExtGetTexturesQCOM(uint* textures,
                                int maxTextures,
                                int* numTextures);

    returns list of texture objects in the current render context.

    The command

        void ExtGetBuffersQCOM(uint* buffers,
                               int maxBuffers,
                               int* numBuffers);

    returns list of buffer objects in the current render context.

    The command

        void ExtGetRenderbuffersQCOM(uint* renderbuffers,
                                     int maxRenderbuffers,
                                     int* numRenderbuffers);

    returns list of render buffer objects in the current render context.

    The command

        void ExtGetFramebuffersQCOM(uint* framebuffers,
                                    int maxFramebuffers,
                                    int* numFramebuffers);

    returns list of frame buffer objects in the current render context.

    The command

        void ExtGetTexLevelParameterivQCOM(uint texture, enum face, int level,
                                           enum pname, int* params);

    returns parameters for texture level chosen with <texture>, <face> and 
    <level>.

    The command

        void ExtTexObjectStateOverrideiQCOM(enum target, enum pname, int param);

    overrides texture parameter for current texture in the <target>. The state 
    chosen with the <pname> argument is restored with <param> value 
    STATE_RESTORE. The STATE_RESTORE has the same effect as calling 
    TexParameteri(enum target, enum pname, int param).

    The command

        void ExtGetTexSubImageQCOM(enum target, int level,
                                   int xoffset, int yoffset, int zoffset,
                                   sizei width, sizei height, sizei depth,
                                   enum format, enum type, void *texels);

    copies texels from current texture in <target> to address in <texels> 
    parameter.

    The command

        void ExtGetBufferPointervQCOM(enum target, void **params)

    returns pointer to the buffer chosen with <target>.

Errors

    INVALID_VALUE error will be generated if the <texture> parameter to
    ExtGetTexLevelParameterivQCOM does not reference to a valid texture object.

    INVALID_ENUM error will be generated if the <texture> parameter to
    ExtGetTexLevelParameterivQCOM does not reference to a valid texture target.

    INVALID_VALUE error will be generated if the <face> parameter to cubemap 
    texture target in ExtGetTexLevelParameterivQCOM is not one of the allowable 
    values.

    INVALID_VALUE error will be generated if the texture <level> parameter to
    ExtGetTexLevelParameterivQCOM is not one of the allowable values.

    INVALID_ENUM error will be generated if the <pname> parameter to
    ExtGetTexLevelParameterivQCOM is not one of the allowable values.

    INVALID_ENUM error will be generated if the <target> parameter to
    ExtGetTexSubImageQCOM is not one of the allowable values.

    INVALID_ENUM error will be generated if the <target> parameter to
    ExtGetTexSubImageQCOM does not reference to a valid texture object.

    INVALID_VALUE error will be generated if the selected region to
    ExtGetTexSubImageQCOM is not inside the texture.

    INVALID_VALUE error will be generated if the <level> parameter to
    ExtGetTexSubImageQCOM does not reference to a valid texture level.

    INVALID_ENUM error will be generated if the <format> or <type> parameters to
    ExtGetTexSubImageQCOM are not one of the allowable values.

    INVALID_ENUM error will be generated if the <target> parameter to
    ExtGetBufferPointervQCOM is not one of the allowable values.

New State

    None.

Revision History

    #01    05/14/2009    Jukka Liimatta       First draft.
