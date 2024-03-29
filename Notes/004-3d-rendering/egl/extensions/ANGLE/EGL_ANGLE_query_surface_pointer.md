# ANGLE_query_surface_pointer

Name

    ANGLE_query_surface_pointer

Name Strings

    EGL_ANGLE_query_surface_pointer

Contributors

    Vladimir Vukicevic
    Daniel Koch

Contacts

    Vladimir Vukicevic (vladimir 'at' pobox.com)

Status

    Complete
    Implemented (ANGLE r558)

Version

    Version 3, February 11, 2011

Number

    EGL Extension #28

Dependencies

    This extension is written against the wording of the EGL 1.4
    Specification. 

Overview

    This extension allows querying pointer-sized surface attributes,
    thus avoiding problems with coercing 64-bit pointers into a 32-bit
    integer.

New Types

    None

New Procedures and Functions

    EGLBoolean eglQuerySurfacePointerANGLE(
                        EGLDisplay dpy,
                        EGLSurface surface,
                        EGLint attribute,
                        void **value);

New Tokens

    None

Additions to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Add to the end of the paragraph starting with "To query an
    attribute associated with an EGLSurface" in section 3.5.6,
    "Surface Attributes":

    "If the attribute type in table 3.5 is 'pointer', then
    eglQuerySurface returns EGL_FALSE and an EGL_BAD_PARAMETER error
    is generated.  To query pointer attributes, call:

         EGLBoolean eglQuerySurfacePointerANGLE(
                             EGLDisplay dpy,
                             EGLSurface surface,
                             EGLint attribute,
                             void **value);

     eglQuerySurfacePointerANGLE behaves identically to eglQuerySurface,
     except that only attributes of type 'pointer' can be queried.
     If an attribute queried via eglQuerySurfacePointerANGLE is not
     of type 'pointer', then eglQuerySurfacePointer returns EGL_FALSE
     and an EGL_BAD_PARAMETER error is generated."

Issues

Revision History

    Version 3, 2011/02/11 - publish

    Version 2, 2010/12/21 - fix typos.

    Version 1, 2010/12/07 - first draft.
