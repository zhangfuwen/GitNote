# OES_mapbuffer

Name

    OES_mapbuffer

Name Strings

    GL_OES_mapbuffer

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

IP Status

    None.

Status

    Ratified by the Khronos BOP, July 22, 2005.

Version

    Last Modifed Date: March 31, 2013

Number

    OpenGL ES Extension #29    

Dependencies

    OpenGL ES 1.1 is required.

Overview

    This extension adds to the vertex buffer object functionality supported
    by OpenGL ES 1.1 or ES 2.0 by allowing the entire data storage of a
    buffer object to be mapped into the client's address space.

Issues

 
New Tokens

    Accepted by the <access> parameter of MapBufferOES:

      WRITE_ONLY_OES               0x88B9

    Accepted by the <value> parameter of GetBufferParameteriv:

      BUFFER_ACCESS_OES            0x88BB
      BUFFER_MAPPED_OES            0x88BC

    Accepted by the <pname> parameter of GetBufferPointervOES:

      BUFFER_MAP_POINTER_OES       0x88BD

New Procedures and Functions

    void GetBufferPointervOES(enum target, enum pname, void** params)

    void *MapBufferOES(enum target, enum access)

    boolean UnmapBufferOES(enum target)

    Please refer to the OpenGL 2.0 specification for details on how
    these functions work.  One departure from desktop OpenGL 2.0 is
    that the <access> parameter to MapBufferOES must be WRITE_ONLY_OES.
    
    Note that any portion of a mapped buffer that is not written
    will retain its original contents.

Errors

    None.

New State

(table 6.8)
                                                  Initial
Get Value              Type  Get Command          Value           Description
---------              ----  -----------          -----           -----------
BUFFER_ACCESS_OES      Z1    GetBufferParameteriv WRITE_ONLY_OES  buffer map flag
BUFFER_MAPPED_OES      B     GetBufferParameteriv FALSE           buffer map flag
BUFFER_MAP_POINTER_OES Y     GetBufferPointervOES NULL            mapped buffer pointer

Revision History

    03/31/2013    Benj Lipchak     Clarify that unwritten data is preserved
    11/12/2007    Benj Lipchak     Mention only WRITE_ONLY access is allowed,
                                   some additional tokens needed
    11/05/2007    Benj Lipchak     Change API version requirement, fix typos
    09/19/2007    Benj Lipchak     Added OES suffixes, GetBufferPointervOES
    07/06/2005    Aaftab Munshi    Created the extension
