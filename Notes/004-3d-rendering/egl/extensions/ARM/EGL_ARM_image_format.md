# ARM_image_format

Name

    ARM_image_format

Name Strings

    EGL_ARM_image_format

Contributors

    Jan-Harald Fredriksen

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

IP Status

    No known IP claims.

Status

    Complete

Version

     Version 1 - February 18, 2020

Number

    138

Dependencies

    This extension is written against the wording of the EGL 1.4
    specification.

    This extension reuses tokens from EGL_EXT_pixel_format_float.

Overview

    When an EGLImage is created from an existing image resource the
    implementation will deduce the format of the image data from that
    resource. In some cases, however, the implementation may not know how to
    map the existing image resource to a known format. This extension extends
    the list of attributes accepted by eglCreateImageKHR such that applications
    can tell the implementation how to interpret the data.

New Procedures and Functions

    None.

New Tokens

   Accepted as an attribute name in the <attrib_list> argument of
   eglCreateImageKHR:
        EGL_COLOR_COMPONENT_TYPE_EXT                   0x3339

   Accepted as attribute values for the EGL_COLOR_COMPONENT_TYPE_EXT attribute
   of eglCreateImageKHR:

        EGL_COLOR_COMPONENT_TYPE_FIXED_EXT              0x333A
        EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT              0x333B
        EGL_COLOR_COMPONENT_TYPE_UNSIGNED_INTEGER_ARM   0x3287
        EGL_COLOR_COMPONENT_TYPE_INTEGER_ARM            0x3288
        EGL_RED_SIZE                                    0x3024
        EGL_GREEN_SIZE                                  0x3023
        EGL_BLUE_SIZE                                   0x3022
        EGL_ALPHA_SIZE                                  0x3021

Modifications to the EGL 1.4 Specification

   Add the following rows to Table 3.xx: Legal attributes for
   eglCreateImageKHR <attrib_list> parameter:

      +------------------------------+------------------------------+-----------+---------------+
      | Attribute                    | Description                  | Valid     | Default Value |
      |                              |                              | <target>s |               |
      +------------------------------+------------------------------+-----------+---------------+
      | EGL_COLOR_COMPONENT_TYPE_EXT | Specifies the component      | All       | NA            |
      |                              | type the EGLImage source     |           |               |
      |                              | is interpreted as            |           |               |
      | EGL_RED_SIZE                 | Specifies the red component  | All       | NA            |
      |                              | size the EGLImage source     |           |               |
      |                              | is interpreted as            |           |               |
      | EGL_GREEN_SIZE               | Specifies the green component| All       | NA            |
      |                              | size the EGLImage source     |           |               |
      |                              | is interpreted as            |           |               |
      | EGL_BLUE_SIZE                | Specifies the blue component | All       | NA            |
      |                              | size the EGLImage source     |           |               |
      |                              | is interpreted as            |           |               |
      | EGL_ALPHA_SIZE               | Specifies the alpha component| All       | NA            |
      |                              | size the EGLImage source     |           |               |
      |                              | is interpreted as            |           |               |
      +------------------------------+------------------------------+-----------+---------------+

    If <attrib_list> specifies values for EGL_COLOR_COMPONENT_TYPE_EXT,
    EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, or EGL_ALPHA_SIZE, the
    implementation will treat these as hints for how to interpret the contents
    of <buffer>.

    EGL_COLOR_COMPONENT_TYPE_EXT indicates the component type of <buffer> and
    must be either EGL_COLOR_COMPONENT_TYPE_FIXED_EXT for fixed-point,
    EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT for floating-point,
    EGL_COLOR_COMPONENT_TYPE_UNSIGNED_INTEGER_ARM for unsigned integer, or
    EGL_COLOR_COMPONENT_TYPE_INTEGER_ARM for integer components.

Add to the list of error conditions for eglCreateImageKHR:

    * If the implementation is unable to interpret the contents <buffer>
      according to the component types and sizes in <attrib_list>, then a
      EGL_BAD_MATCH error is generated.

Issues

    1. Should there be a way to specify the component order?

       Resolved. No, the component order is interpreted to be R, G, B, A,
       with R mapping to component 0. If the application needs a different
       component order it can use swizzle in the client API side or in the
       shader.

Revision History

    Version 1, 2020/02/18
      - Internal revisions
