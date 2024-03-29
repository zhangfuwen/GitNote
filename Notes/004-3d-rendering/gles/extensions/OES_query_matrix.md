# OES_query_matrix

Name

    OES_query_matrix

Name Strings

    GL_OES_query_matrix

Contact

    Kari Pulli, Nokia (kari.pulli 'at' nokia.com)

Notice

    Copyright (c) 2003-2013 The Khronos Group Inc. Copyright terms at
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

    Ratified by the Khronos BOP, July 23, 2003.

Version

    $Date: 2003/07/23 04:23:25 $ $Revision: 1.2 $

Number

    OpenGL ES Extension #16 (formerly OpenGL Extension #296)

Dependencies

    OpenGL 1.3 is required.
    OES_fixed_point is required.

Overview

    Many applications may need to query the contents and status of the
    current matrix at least for debugging purposes, especially as the
    implementations are allowed to implement matrix machinery either in
    any (possibly proprietary) floating point format, or in a fixed point
    format that has the range and accuracy of at least 16.16 (signed 16 bit
    integer part, unsigned 16 bit fractional part).
    
    This extension is intended to allow application to query the components
    of the matrix and also their status, regardless whether the internal
    representation is in fixed point or floating point.
     
IP Status

    There is no intellectual property associated with this extension.

Issues

    None known.

New Procedures and Functions

    GLbitfield glQueryMatrixxOES( GLfixed mantissa[16],
                                  GLint   exponent[16] )

    mantissa[16] contains the contents of the current matrix in GLfixed
    format.  exponent[16] contains the unbiased exponents applied to the
    matrix components, so that the internal representation of component i
    is close to mantissa[i] * 2^exponent[i].  The function returns a status
    word which is zero if all the components are valid. If
    status & (1<<i) != 0, the component i is invalid (e.g., NaN, Inf).
    The implementations are not required to keep track of overflows.  In
    that case, the invalid bits are never set.

New Tokens

    None

Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL 1.3 Specification (Per-Fragment
Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL 1.3 Specification (State and
State Requests)

    Insert Overview and New Procedures and Functions to become Section 6.1.13.

Additions to Appendix A of the OpenGL 1.3 Specification (Invariance)

    None

Additions to the AGL/GLX/WGL Specifications

GLX Protocol

    QueryMatrixxOES() is mapped to the equivalent protocol for
    floating-point state queries.  Two queries are required; one to
    retrieve the current matrix mode and another to retrieve the
    matrix values.

Dependencies on OES_fixed_point

    OES_fixed_point is required for the GLfixed definition.

Errors

    None

New State

    None

New Implementation Dependent State

    None

Revision History

Apr 15, 2003    Kari Pulli      Created the document
Jul 08, 2003    David Blythe    Clarified the Dependencies section,
                                Added extension number
Jul 12, 2003    David Blythe    Add GLX protocol note

