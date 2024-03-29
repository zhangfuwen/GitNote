# EXT_EGL_image_external_wrap_modes

Name

    EXT_EGL_image_external_wrap_modes

Name Strings

    GL_EXT_EGL_image_external_wrap_modes

Contributors

    Jeff Leger, Qualcomm
    Rob VanReenen, Qualcomm
    Jonathan Wicks, Qualcomm
    John Carmack, Oculus
    Cass Everitt, Oculus
    Graeme Leese, Broadcom

Contacts

    Jeff Leger, Qualcomm  (jleger 'at' qti.qualcomm.com)

Status

    Complete

Version

    Last Modified Date: Feb 06, 2018
    Revision: #4

Number

    OpenGL ES Extension #298

Dependencies

    Requires OES_EGL_image_external.

    Interacts with OES_EGL_image_external_essl3.

    OES_texture_border_clamp affects the definition of this extension.

    The portions of this extension that modify/extend 
    OES_EGL_image_external are written against OpenGL ES 2.0.
    The portions of this extension that modify/extend 
    OES_texture_border_clamp are written against OpenGL ES 3.0.
    The portions of this extension that modify/extend OES_EGL_image_external_essl3 are
    written against OpenGL ES 3.0.

Overview

    This extension builds on OES_EGL_image_external, which only allows
    a external images to use a single clamping wrap mode:  CLAMP_TO_EDGE.
    This extension relaxes that restriction, allowing wrap modes REPEAT
    and MIRRORED_REPEAT.  If OES_texture_border_clamp is supported, then 
    CLAMP_TO_BORDER is also allowed.

    This extension similarly adds to the capabilities of OES_EGL_image_external_essl3,
    allowing the same additional wrap modes.

    Since external images can be non-RGB, this extension clarifies how
    border color values are specified for non-RGB external images.

IP Status

    No known IP claims.

New Procedures and Functions

    None.

New Types

    None.

New Tokens

     None.

Changes to Chapter 3 of the OpenGL ES 2.0 Specification

    Modify the fourth sentence of the first paragraph of Section 3.7.14,
    as added by OES_EGL_image_external
    
    from:
         "The default s and t wrap modes are CLAMP_TO_EDGE and it is an
         INVALID_ENUM error to set the wrap mode to any other value."
    to:
        [[ The following applies if OES_texture_border_clamp is not supported. ]]

        "The default s and t wrap modes are CLAMP_TO_EDGE and it is an
        INVALID_ENUM error to set the wrap mode to any value other than
        CLAMP_TO_EDGE, REPEAT, or MIRRORED_REPEAT."

        [[ The following applies if OES_texture_border_clamp is supported. ]]

        "The default s and t wrap modes are CLAMP_TO_EDGE and it is an
        INVALID_ENUM error to set the wrap mode to any value other than
        CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT, or CLAMP_TO_BORDER."

    [[ The following applies if OES_texture_border_clamp is supported. ]]

    Add the following paragraph to the end of Section 3.7.14, as added
    by OES_EGL_image_external:

        The state TEXTURE_BORDER_COLOR_OES is specified as an RGBA color
        in linear color space.  For example, if the original image is stored
        in(non-linear) ITU-R Rec. 601 YV12, the TEXTURE_BORDER_COLOR_OES
        must still be specified as an RGBA color in linear color space.

Changes to section "3.7.4 Texture Parameters" of the OpenGL ES 2.0 
Specification

    Modify the paragraph as added by OES_EGL_image_external
    
    from:
            "When <target> is TEXTURE_EXTERNAL_OES only NEAREST and 
        LINEAR are accepted as TEXTURE_MIN_FILTER and only CLAMP_TO_EDGE
        is accepted as TEXTURE_WRAP_S and TEXTURE_WRAP_T."
    to:
        [[ The following applies if OES_texture_border_clamp is not supported. ]]

            "When <target> is TEXTURE_EXTERNAL_OES only NEAREST and
        LINEAR are accepted as TEXTURE_MIN_FILTER and only CLAMP_TO_EDGE,
        REPEAT, or MIRRORED_REPEAT are accepted as TEXTURE_WRAP_S and
        TEXTURE_WRAP_T."

        [[ The following applies if OES_texture_border_clamp is supported. ]]

            "When <target> is TEXTURE_EXTERNAL_OES only NEAREST and
        LINEAR are accepted as TEXTURE_MIN_FILTER and only CLAMP_TO_EDGE,
        REPEAT, MIRRORED_REPEAT, or CLAMP_TO_BORDER are accepted as 
        TEXTURE_WRAP_S and TEXTURE_WRAP_T."

[[ The following applies if OES_texture_border_clamp is supported. ]]

Changes to section 3.8.10 "Texture Minification" of the OpenGL ES 3.0
Specification

    Modify the sentence added by OES_texture_border_clamp

    From:
        "If the texture contains color components, the values of 
        TEXTURE_BORDER_COLOR_OES are interpreted as an RGBA color to 
        match the texture's internal format in a manner consistent with
        table 3.11."

    To:
         "If the texture contains color components, the values of
        TEXTURE_BORDER_COLOR_OES are interpreted as an RGBA color in
        linear color space to match the texture's internal format in a
        manner consistent with table 3.11, except that if an external
        texture stores YUV values then the linear RGBA border value is
        first converted into a YUVA value in the colorspace of the
        texture."

[[ The following applies if OES_EGL_image_external_essl3 is supported. ]]

Changes to section 3.8.2 "Sampler Objects" of the OpenGL ES 3.0.2
Specification

    Modify the the following sentence added by OES_EGL_image_external_essl3

    From:
        "For example, if TEXTURE_WRAP_S or TEXTURE_WRAP_T is set to
         anything but CLAMP_TO_EDGE on the sampler object bound to a
         texture unit and the texture bound to that unit is an external
         texture, the texture will be considered incomplete."

    To:

        [[ The following applies if OES_texture_border_clamp is not supported. ]]

         "For example, if TEXTURE_WRAP_S or TEXTURE_WRAP_T is set to
          anything but CLAMP_TO_EDGE, REPEAT, or MIRRORED_REPEAT on the
          sampler object bound to a texture unit and the texture bound
          to that unit is an external texture, the texture will be
          considered incomplete."

        [[ The following applies if OES_texture_border_clamp is supported. ]]

         "For example, if TEXTURE_WRAP_S or TEXTURE_WRAP_T is set to
          anything but CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT, or
          CLAMP_TO_BORDER on the sampler object bound to a texture unit
          and the texture bound to that unit is an external texture,
          the texture will be considered incomplete."

Issues

    1) For YUV texture formats, should the should the border color be 
    specified as RGBA or YUVA ?

    Resolved:  The border color should be specified as linear RGBA since the 
    application may not know the underlying texture format/colorspace.  The 
    color should be converted by the implementation to a colorspace (e.g., 
    ITU-R Rec. 601) matching the texture's internal format.


Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    12/19/17   jwicks    Initial spec
     2.   01/03/18   jleger    Updates and cleanup.
     3.   01/05/18   jleger    Allow additional wrap modes.  Rename the extension.
     4.   02/06/18   jleger    Added interactions with OES_EGL_image_external_essl3.
