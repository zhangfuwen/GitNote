# OES_extended_matrix_palette

Name

    OES_extended_matrix_palette

Name Strings

    GL_OES_extended_matrix_palette

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2005-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Ratified by the Khronos BOP, July 22, 2005.

Version

    Last Modified Date: February 3, 2005

Number

    OpenGL ES Extension #8

Dependencies

    OES_matrix_palette is required
    OpenGL ES 1.1 is required.

Overview

    The OES_matrix_palette extension added the ability to support vertex skinning 
    in OpenGL ES.  One issue with OES_matrix_palette is that the minimum size of 
    the matrix palette is very small.  This leads to applications having to break 
    geometry into smaller primitive sets called via. glDrawElements.  This has an 
    impact on the overall performance of the OpenGL ES implementation.  In general, 
    hardware implementations prefer primitive packets with as many triangles as 
    possible.  The default minimum size defined in OES_matrix_palette is not 
    sufficient to allow this.  The OES_extended_matrix_palette extension increases 
    this minimum from 9 to 32.  

        Another issue is that it is very difficult for ISVs to handle different 
        size matrix palettes as it affects how they store their geometry 
        in the database - may require multiple representations which is
        not really feasible.  So the minimum size is going to be what most ISVs
        will use.

        By extending the minimum size of the matrix palette, we remove this
        fragmentation and allow applications to render geometry with minimal
        number of calls to glDrawElements or glDrawArrays.  The OpenGL ES
        implementation can support this without requiring any additional hardware
        by breaking the primitive, plus it gives implementations the flexibility
        to accelerate with a bigger matrix palette if they choose to do so.

        Additionally, feedback has also been received to increase the number of
        matrices that are blend per vertex from 3 to 4.  The OES_extended_matrix_palette
        extension increases the minium number of matrices / vertex to 4.
    
IP Status

    None. 

Issues

    None

New Procedures and Functions

    None

New Tokens

    No new tokens added except that the default values for
    MAX_PALETTE_MATRICES_OES and MAX_VERTEX_UNITS_OES are 32 and 4 respectively.

Additions to Chapter 2 of the OpenGL ES 1.0 Specification

    None

Errors

    None

New State

Get Value                   Type  Command      Value    Description 
---------                   ----  -------      -------  -----------

MAX_PALETTE_MATRICES_OES    Z+    GetIntegerv  32       size of matrix palette
MAX_VERTEX_UNITS_OES        Z+    GetIntegerv  4        number of matrices per vertex

Revision History

    Feb 03, 2005   Aaftab Munshi    First draft of extension
