# APPLE_texture_max_level

Name

    APPLE_texture_max_level

Name Strings

    GL_APPLE_texture_max_level

Contributors

    Contributors to SGIS_texture_lod desktop OpenGL extension from which 
    this extension borrows heavily.

Contacts

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Last Modified Date: February 24, 2011
    Revision: #2

Number

    OpenGL ES Extension #80

Dependencies

    Written based on the wording of the OpenGL ES 2.0 specification.

    OpenGL ES 1.1 affects the definition of this extension.

Overview

    This extension allows an application to specify the maximum (coarsest) 
    mipmap level that may be selected for the specified texture.  This maximum
    level is also used to determine which mip levels are considered when 
    determining texture completeness.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameter of TexParameteri, TexParameterf,
    TexParameteriv, TexParameterfv, GetTexParameteriv, and GetTexParameterfv:

        TEXTURE_MAX_LEVEL_APPLE          0x813D

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Add the following line to Table 3.10 (Texture parameters and their values):

        Name                            Type        Legal Values
        ----                            ----        ------------
        TEXTURE_MAX_LEVEL_APPLE         integer     any non-negative integer

    In section 3.7.4, insert the following paragraph before the paragraph
    beginning "Texture parameters for a cube map texture...":
    
    "In the remainder of section 3.7, denote by level_max the value of the 
    texture parameter TEXTURE_MAX_LEVEL_APPLE."

    In section 3.7.7 subsection "Mipmapping" modify the second paragraph
    such that p and q are defined as follows:
    
        p = floor(log2(max(w_b, h_b)))
        q = min{p, level_max}

    In section 3.7.7 subsection "Mipmapping" insert the following paragraph 
    before the paragraph beginning "The mipmap is used in conjunction...":
    
    "The value of level_max may be respecified for a specific texture by
    calling TexParameter[if] with <pname> set to TEXTURE_MAX_LEVEL_APPLE.
    The error INVALID_VALUE is generated if the specified value is negative."

    In section 3.7.12, modify the last three sentences to read as follows:
    
    "Next, there are the two sets of texture properties; each consists of the
    selected minification and magnification filters, the maximum array level,
    and the wrap modes for s and t.  In the initial state, the value assigned
    to TEXTURE_MIN_FILTER is NEAREST_MIPMAP_LINEAR, and the value for 
    TEXTURE_MAG_FILTER is LINEAR.  s and t wrap modes are both set to REPEAT.
    The value of TEXTURE_MAX_LEVEL_APPLE is 1000."

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment 
Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State 
Requests)

    None

Dependencies on OpenGL ES 1.1

    On an OpenGL ES 1.1 implementation, include fixed-point flavors of
    TexParameter and GetTexParameter commands.
    
Errors

    INVALID_VALUE is generated if an attempt is made to set
    TEXTURE_MAX_LEVEL_APPLE to a negative value.

New State

    Add the following to Table 6.8 (Textures (state per texture object):
                                                           Initial
    Get Value                 Get Command         Type     Value     Description       Sec.
    ---------                 -----------------   ------   -------   -----------       ----
    TEXTURE_MAX_LEVEL_APPLE   GetTexParameteriv   n x Z+   1000      Maximum texture   3.7
                                                                     array level

New Implementation Dependent State

    None

Revision History

    #2  02/24/2011    Benj Lipchak     Assign extension number
    #1  10/29/2009    Benj Lipchak     First draft
