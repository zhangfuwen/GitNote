# NV_bgr

Name

    NV_bgr

Name Strings

    GL_NV_bgr

Contributors

    Koji Ashida, NVIDIA
    Greg Prisament, NVIDIA
    Greg Roth, NVIDIA

Contact

    Greg Roth (groth 'at' nvidia.com)

Version

    Last Modified: 07 Jan 2013
    NVIDIA Revision: 3

Number

    OpenGL ES Extension #135

Status
    
    Complete.

Dependencies

    OpenGL ES 2.0 is required.

    Written against the OpenGL ES 2.0.25 specification

    This extension interacts trivially with NV_texture_array.

Overview

    NV_bgr extends the list of color formats used in texture images
    and reading pixels. Specifically, it adds color formats with BGR
    ordering of color channels.

New Tokens

    Accepted by the <internalformat> and <format> parameter of
    ReadPixels, TexImage2D, TexSubImage2D, TexImage3DNV, and 
    TexSubImage3DNV:

        BGR_NV                  0x80E0

Changes to Chapter 3 of the OpenGL ES 2.0.25 Specification (Rasterization)

    Changes to Section 3.6.2 "Transfer of Pixel Rectangles"

    Add the following entries to Table 3.3 "TexImage2D and ReadPixels
    formats":

        Format Name  Element Meaning and Order   Target Buffer
        -----------  --------------------------  ---------------
        BGR_NV       B,G,R                       Color

    Add the following entries to Table 3.4 "Valid pixel format and type
    combinations":

        Internal
        Format    Type                    Bytes per Pixel
        --------  ----------------------  ---------------
        BGR_NV    UNSIGNED_BYTE                  3

Interactions with NV_texture_array

    If NV_texture_array is not supported, ignore references to
    TexImage3DNV and TexSubImage3DNV.

Revision History

    Rev.    Date          Author       Changes
    ----   ------------   ---------    -------------------------------------
     3     07 Jan 2013    groth        Fix minor suffix mistake
     2     23 Oct 2012    groth        Formatting changed. Additional tables.
     1     03 June 2008   kashida      First draft written based on EXT_bgra.

