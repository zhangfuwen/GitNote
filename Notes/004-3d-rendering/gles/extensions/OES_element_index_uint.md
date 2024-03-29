# OES_element_index_uint

Name

    OES_element_index_uint

Name Strings

    GL_OES_element_index_uint

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

    Last Modifed Date: November 5, 2007

Number

    OpenGL ES Extension #26    

Dependencies

    OpenGL ES 1.0 is required.

Overview

    OpenGL ES 1.0 supports DrawElements with <type> value of
    UNSIGNED_BYTE and UNSIGNED_SHORT.  This extension adds
    support for UNSIGNED_INT <type> values.

Issues

 
New Tokens

    Accepted by the <type> parameter of DrawElements:

        UNSIGNED_INT                0x1405

New Procedures and Functions

    None.

Errors

    None.

New State

    None.

Revision History

    11/05/2007    Benj Lipchak     Change API version requirement
    07/06/2005    Aaftab Munshi    Created the extension
