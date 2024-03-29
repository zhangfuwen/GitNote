# EXT_pixel_format_float

Name

    EXT_pixel_format_float

Name Strings

    EGL_EXT_pixel_format_float

Contributors

    Tom Cooksey
    Jesse Hall
    Mathias Heyer
    Adam Jackson
    James Jones
    Daniel Koch
    Jeff Leger
    Weiwan Liu
    Jeff Vigil

Contact

    Weiwan Liu, NVIDIA (weiwliu 'at' nvidia.com)

Status

    Complete

Version

    Version 4 - Nov 22, 2016

Number

    EGL Extension #106

Dependencies

    This extension is written against the wording of the EGL 1.5 specification
    (August 27, 2014).

Overview

    This extensions aims to provide similar functionality as GL_ARB_color_-
    buffer_float, WGL_ARB_pixel_format_float and GLX_ARB_fbconfig_float. This
    extension allows exposing new EGLConfigs that support formats with
    floating-point RGBA components. This is done by introducing a new EGLConfig
    attribute that represents the component type, i.e. fixed-point or
    floating-point. Such new EGLConfigs can be used to create floating-point
    rendering surfaces and contexts.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute name in the <attrib_list> argument of
    eglChooseConfig, and the <attribute> argument of eglGetConfigAttrib:

        EGL_COLOR_COMPONENT_TYPE_EXT              0x3339

    Accepted as attribute values for the EGL_COLOR_COMPONENT_TYPE_EXT attribute
    of eglChooseConfig:

        EGL_COLOR_COMPONENT_TYPE_FIXED_EXT        0x333A
        EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT        0x333B

    Additions to table 3.1, "EGLConfig attributes" in Section 3.4 "Configuration
    Management":

        Attribute                       Type       Notes
        ---------                       ----       ---------
        EGL_COLOR_COMPONENT_TYPE_EXT     enum       color component type

    Append one paragraph at the end of "The Color Buffer" section on page 21:

        EGL_COLOR_COMPONENT_TYPE_EXT indicates the color buffer component type,
        and must be either EGL_COLOR_COMPONENT_TYPE_FIXED_EXT for fixed-point
        color buffers, or EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT for floating-point
        color buffers.

    Add one entry to Table 3.4 and increment "Sort Priority" between "2" and
    "11" by one for existing entries:

        Attribute                      Default
        -----------                    ------------
        EGL_COLOR_COMPONENT_TYPE_EXT    EGL_COLOR_COMPONENT_TYPE_FIXED_EXT

        Selection Criteria    Sort Order    Priority
        ------------------    ----------    --------
        Exact                 Special        2

    Insert before the entry for EGL_COLOR_BUFFER_TYPE, and increment its
    numbering and subsequent numbering by one:

        2. Special: by EGL_COLOR_COMPONENT_TYPE_EXT where the precedence is
        EGL_COLOR_COMPONENT_TYPE_FIXED_EXT, EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT

    Change footnote 8 on page 30 to:

        Within the same EGL_COLOR_COMPONENT_TYPE_EXT, this rule places configs
        with deeper color buffers first in the list returned by
        eglChooseConfig...

Issues

    1. When reading from or rendering to a floating-point EGL surface, is there
       any clamping performed on the values?

       RESOLVED: It depends on the behavior of the client API. For example, in
       OpenGL and ES, by default no clamping will be done on the floating-point
       values, unless the clamping behavior is changed via the client API.

    2. When rendering to a floating-point EGL surface, since values may not be
       clamped to [0, 1], what is the range of values that applications can use
       to get display's "darkest black" and "brightest white"?

       RESOLVED: It is not in the scope of this extension to define a range of
       values that corresponds to display's capability. Please refer to the EGL
       specification for the chosen colorspace (EGL_GL_COLORSPACE), where such a
       reference range may be defined.

Revision History

    Rev.     Date        Author          Changes
    ----   --------  ---------------  ------------------------------------------
     1     12/11/15   Weiwan Liu      Initial version
     2     05/18/16   Weiwan Liu      Rename to EXT
     3     05/31/16   Weiwan Liu      Add issues
     4     11/22/16   Weiwan Liu      Change status to complete

