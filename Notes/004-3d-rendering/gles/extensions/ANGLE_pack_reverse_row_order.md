# ANGLE_pack_reverse_row_order

Name

    ANGLE_pack_reverse_row_order

Name Strings

    GL_ANGLE_pack_reverse_row_order

Contact

    Daniel Koch, TransGaming (daniel 'at' transgaming.com)

Contributors

    Brian Salomon
    Daniel Koch

Status

    Implemented in ANGLE ES2

Version

    Last Modified Date: February 22, 2011
    Author Revision: 22

Number

    OpenGL ES Extension #110

Dependencies

    OpenGL 1.5 or OpenGL ES 1.0 are required.

    Some of the functionality of this extension is not supported
    when implemented against OpenGL ES.

    EXT_texture_rg interacts with this extension.

    The extension is written against the OpenGL 3.2 Specification
    (Core Profile).

Overview

    This extension introduces a mechanism to allow reversing the order
    in which image rows are written into a pack destination. This
    effectively allows an application to flip the results of a ReadPixels
    in the y direction operation without having to render upside down.

    The coordinate system of OpenGL is vertically reversed in comparison to a
    number of other graphics systems such as native windowing APIs. Applications
    that perform ReadPixels may have to either render to an intermediate color
    buffer before calling ReadPixels or perform a flip in software after
    ReadPixels. In some systems the GL can perform the row reversal during
    ReadPixels without incurring additional cost.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Types

    None

New Tokens

    Accepted by the <pname> parameter of PixelStore{if}, GetIntegerv(),
    GetBooleanv(), and GetFloatv():

        PACK_REVERSE_ROW_ORDER_ANGLE    0x93A4

Additions to Chapter 3 of the OpenGL 3.2 Specification (Rasterization)

    In Section 4.3.1 (Reading Pixels) add a row to table 4.7:

        +------------------------------+---------+---------------+-------------+
        | Parameter Name               | Type    | Initial Value | Valid Range |
        +------------------------------+---------+---------------+-------------+
        | PACK_REVERSE_ROW_ORDER_ANGLE | boolean | FALSE         | TRUE/FALSE  |
        +------------------------------+---------+---------------+-------------+

    In Section 4.3.1 (Reading Pixels) modify the second paragraph of subsection 
    "Placement in Pixel Pack Buffer or Client Memory" to read:

        When PACK_REVERSE_ROW_ORDER_ANGLE is FALSE groups of elements are placed
        in memory just as they are taken from memory when transferring pixel
        rectangles to the GL. That is, the ith group of the jth row
        (corresponding to the ith pixel in the jth row) is placed in memory just
        where the ith group of the jth row would be taken from when transferring
        pixels. See Unpacking under section 3.7.2. The only difference is that
        the storage mode parameters whose names begin with PACK_ are used
        instead of those whose names begin with UNPACK_. If the format is RED,
        GREEN, BLUE, or ALPHA, only the corresponding single element is written.
        Likewise if the format is RG, RGB, or BGR, only the corresponding two or
        three elements are written. Otherwise all the elements of each group are
        written. When PACK_REVERSE_ROW_ORDER_ANGLE is TRUE the order of the rows
        of elements is reversed before the data is packed. That is, the element
        corresponding to pixel (x, y + height - 1) becomes the first element
        packed, followed by (x + 1, y + height - 1), etc. Otherwise, pixel data
        is packed in the same manner as when PACK_REVERSE_ROW_ORDER_ANGLE is
        FALSE.

Additions to Chapter 6 of the OpenGL 3.2 Specification (State and State Requests)

    In Section 6.1.4 add the following sentence to the fifth paragraph
    (beginning with "For three-dimensional and two-dimensional array
    textures..."):
        When PACK_REVERSE_ROW_ORDER_ANGLE is TRUE the order of rows within
        each image are reversed without reordering the images themselves.

Dependencies on OpenGL ES

    If implemented for OpenGL ES, this extension behaves as specified, except:

    -Delete all references to formats RED, GREEN, BLUE, RG, and BGR.

    -The language about image order in Section 6.1.4 does not apply as OpenGL ES
     does not have GetTexImage.

Dependencies on EXT_texture_rg

    If EXT_texture_rg is present reinsert language about formats RED and RG
    into the OpenGL ES 2.0 specification.

Errors

    None

New State
                                                           Initial
    Get Value                       Type  Get Command      Value    Description                    Sec.
    ---------                       ----  -----------      -------  -----------                    ----
    PACK_REVERSE_ROW_ORDER_ANGLE    B     GetIntegerv      FALSE    Pixel pack row order reversal  4.3.1

New Implementation Dependent State

    None

Issues

    None

Sample Code

    /* Allocate space to hold the pixel data */
    const GLvoid* pixels = malloc(width * height * 4);

    /* Bind the framebuffer object to be read */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    /* Enable row order reversal */
    glPixelStore(GL_PACK_REVERSE_ROW_ORDER_ANGLE, TRUE);

    /* The pixel data stored in pixels will be in top-down order, ready for
     * use with a windowing system API that expects this order.
     */
    glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

Revision History

    Revision 1, 2011/11/22 (Brian Salomon)
      - First version
    Revision 2, 2012/02/22 (dgkoch)
      - prepare for publishing
