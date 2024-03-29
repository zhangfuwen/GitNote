# KHR_texture_compression_astc_sliced_3d

Name

    KHR_texture_compression_astc_sliced_3d

Name Strings

    GL_KHR_texture_compression_astc_sliced_3d

Contact

    Eric Werness (ewerness 'at' nvidia.com)

Contributors

    Sean Ellis, ARM
    Jorn Nystad, ARM
    Tom Olson, ARM
    Andy Pomianowski, AMD
    Cass Everitt, NVIDIA
    Walter Donovan, NVIDIA
    Robert Simpson, Qualcomm
    Maurice Ribble, Qualcomm
    Larry Seiler, Intel
    Daniel Koch, NVIDIA
    Anthony Wood, Imagination Technologies
    Jon Leech

IP Status

    No known issues.

Notice

    Copyright (c) 2015 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL and OpenGL ES Working Groups. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete.
    Approved by the OpenGL ES Working Group on 2015/07/22
    Approved by the OpenGL ARB Working Group on 2015/07/31
    Ratified by the Khronos Board of Promoters on 2015/10/09.

Version

    Version 2, September 15, 2015

Number

    ARB Extension #189
    OpenGL ES Extension #249

Dependencies

    Written based on the wording of the OpenGL ES 3.1 (April 29, 2015)
    Specification

    Requires GL_KHR_texture_compression_astc_ldr

Overview

    Adaptive Scalable Texture Compression (ASTC) is a new texture
    compression technology that offers unprecendented flexibility, while
    producing better or comparable results than existing texture
    compressions at all bit rates. It includes support for 2D and
    slice-based 3D textures, with low and high dynamic range, at bitrates
    from below 1 bit/pixel up to 8 bits/pixel in fine steps.

    This extension extends the functionality of
    GL_KHR_texture_compression_astc_ldr to include slice-based 3D textures
    for textures using the LDR profile in the same way as the HDR profile
    allows slice-based 3D textures.

Interactions

    None

New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 8 of the OpenGL ES 3.1 Specification (Textures and Samplers)

    Modify table 8.19 (Compressed internal formats), as modified by
    GL_KHR_texture_compression_astc_ldr

    Modify the "3D Tex." column to be checked for all ASTC formats.

Additions to Appendix C of the OpenGL ES 3.1 Specification (Compressed
Texture Image Formats

    Modify the sub-section on ASTC image formats, C.2.25 LDR PROFILE SUPPORT

    Change the first bullet on the feature subset list to read.

    * 2D and slice-based 3D textures only, including 2D, 2D array, cube map
        face, cube map array, and 3D texture targets.

Revision History

    Revision 2, September 15, 2015 (Jon Leech) - correct typo from
    "GL_KHR_texture_compression_ldr" to
    "GL_KHR_texture_compression_astc_ldr"

    Revision 1, June 18, 2015 - initial revision.
