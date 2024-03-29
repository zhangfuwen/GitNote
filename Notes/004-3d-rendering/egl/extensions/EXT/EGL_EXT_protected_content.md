# EXT_protected_content

Name

    EXT_protected_content

Name Strings

    EGL_EXT_protected_content

Contributors

    Ramesh Viswanathan
    Brian Ellis
    Colin Sharp
    Rajeev Kulkarni
    Mohan Maiya
    Maurice Ribble
    Craig Donner
    Jan-Harald Fredriksen
    Daniel Koch
    Michael Golds
    Ray Smith

Contacts

    Maurice Ribble (mribble 'at' qti.qualcomm.com)

IP Status

    No known IP claims.

Status

    Complete.

Version

    Version 13, December 6, 2021

Number

    EGL Extension #97

Dependencies

    Requires EGL 1.4.

    Interactions with EGL_KHR_image_base extension.

    This extension is written against the wording of the EGL 1.4.
    Specification (12/04/2013)

    This extension has interactions with EGL_EXT_protected_surface if that
    extension is supported.  The interactions are described in the main text.

Overview

    This extension introduces the concept of protected contexts and protected
    resources, specifically surfaces and EGLImages. Applications can choose at
    creation time whether a context, surface or EGLImage is protected or not.

    A protected context is required to allow the GPU to operate on protected
    resources, including protected surfaces and protected EGLImages.

    An explanation of undefined behavior in this extension: Several places
    in this extension mention undefined behavior can result, which can
    include program termination. The reason for this is because one way
    to handle protected content is by using a protected virtual to physical
    memory translation layer. With this sort of solution a system may generate
    read or write faults when a non-protected source tries to access a protected
    buffer. Depending on the system these faults might be ignored or they might
    cause process termination. This undefined behavior should not include
    actually allowing a transfer of data from a protected surface to a
    non-protected surface.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute name in the <attrib_list> parameter of
    eglCreateContext, eglCreateWindowSurface, and eglCreateImageKHR;
    and as an parameter of eglQuerySurface and eglQueryContext:

        EGL_PROTECTED_CONTENT_EXT               0x32C0
        

Add a new section 2.7 entitled "Protected Content" at the end of Chapter 2 (EGL
Operation)

   "The attribute EGL_PROTECTED_CONTENT_EXT can be applied to EGL contexts,
    EGL surfaces and EGLImages. If the attribute EGL_PROTECTED_CONTENT_EXT
    is set to EGL_TRUE by the application, then the newly created EGL object
    is said to be protected. A protected context is required to allow the
    GPU to operate on protected resources, including protected surfaces and
    protected EGLImages.

    GPU operations are grouped into pipeline stages. Pipeline stages can be
    defined to be protected or not protected. Each stage defines
    restrictions on whether it can read or write protected and unprotected
    resources, as follows:

    When a GPU stage is protected, it:
    - Can read from protected resources
    - Can read from unprotected resources
    - Can write to protected resources
    - Can NOT write to unprotected resources

    When a GPU stage is not protected, it:
    - Can NOT read from protected resources
    - Can read from unprotected resources
    - Can NOT write to protected resources
    - Can write to unprotected resources

    Any accesses not following these restrictions will result in undefined
    behavior.

    This extension does not specify which pipeline stages of a protected
    context are protected or not. This is left to a client API extension to
    define. All stages in a regular (not protected) context are not
    protected. However, if EGL_EXT_protected_surface is also supported, a
    regular (not protected) context will execute stages where one or more
    protected resources is accessed as if it were a protected context.

    Note that the protection state of a stage may be left implementation
    defined by a client API extension. This means that no guarantees can be
    made about whether the stage will be protected or not protected.
    Practically this means that the permitted operations for such a stage
    are the intersection of the allowed operations for protected and not
    protected stages, i.e it:

    - Can NOT read from protected resources
    - Can read from unprotected resources
    - Can NOT write to protected resources
    - Can NOT write to unprotected resources

    Since this is not a very useful set of operations refer to the client API
    extension to see what operations are actually allowed.

    This extension does not guarantee the implementation abides by a
    system's digital rights management requirements. It must be verified
    beyond the existence of this extension that the implementation of this
    extension is trustworthy according to the requirements of a content
    protection system."

Additions to Chapter 3 of the EGL 1.4 Specification (Rendering Contexts)

    Change the fifth paragraph in section 3.7.1 Creating Rendering Contexts:

        "attrib list specifies a list of attributes for the context. The
        list has the same structure as described for eglChooseConfig.
        Attributes that can be specified in attrib list include
        EGL_CONTEXT_CLIENT_VERSION and EGL_PROTECTED_CONTENT_EXT. The
        EGL_CONTEXT_CLIENT_VERSION attribute may only be specified when
        creating a OpenGL ES context (e.g. when the current rendering API is
        EGL_OPENGL_ES_API)."

    Add the following paragraph in section 3.7.1 on p. 44 before "attrib list
    may be NULL or empty (first attribute is EGL_NONE), in which case
    attributes assume their default values as described below."

       "EGL_PROTECTED_CONTENT_EXT specifies the protected state of the new
        context. If its value is EGL_TRUE, then the context is said to be
        protected. If its value is EGL_FALSE, then the context is not
        protected. See section 2.7 (Protected Content) for more information
        about protected contexts.

        The default value of EGL_PROTECTED_CONTENT_EXT is EGL_FALSE."

    Add the following paragraph in section 3.7.4 Context Queries. Add after
    the last paragraph after eglQueryContext queries.
    
        "Querying EGL_PROTECTED_CONTENT_EXT returns the current value"

Additions to Chapter 3 of the EGL 1.4 Specification (Rendering Surfaces)

    Change the second paragraph in section 3.5 on p. 28 (describing
    eglCreateWindowSurface):

        "Attributes that can be specified in attrib list include
        EGL_RENDER_BUFFER, EGL_PROTECTED_CONTENT_EXT, EGL_VG_COLORSPACE, and
        EGL_VG_ALPHA_FORMAT."

    Add the following paragraph in section 3.5 on p. 28 before
    "EGL_VG_COLORSPACE specifies the color space used by OpenVG" (describing
    eglCreateWindowSurface(attrib_list):

        "EGL_PROTECTED_CONTENT_EXT specifies the protected state of the
        window surface. If its value is EGL_TRUE, then the surface content
        is said to be protected. If its value is EGL_FALSE, then the surface
        content is not protected. See section 2.7 (Protected Content) for
        more information about protected and non-protected surfaces.

        Client APIs will not allow contents of protected surfaces to be
        accessed by non-protected contexts in the system (including
        non-secure software running on the CPU). Such operations will result
        in undefined behavior.

        Calling eglSwapBuffers on such a protected surface will succeed, but
        the contents may or may not be posted successfully depending on
        whether those parts of the pipeline are capable of handling
        protected content. Any disallowed operation will fail and result in
        undefined behavior.

        The default value of EGL_PROTECTED_CONTENT_EXT is EGL_FALSE."

    Add the following paragraph in section 3.5.6 Surface Attributes. Add after
    the last paragraph after eglQuerySurface attribute queries.
    
        "Querying EGL_PROTECTED_CONTENT_EXT returns the current value"

Additions to EGL_KHR_image_base extension specification

    Add to section 2.5.1 Table bbb:
      +-----------------------------+-------------------------+---------------+
      | Attribute                   | Description             | Default Value |
      +-----------------------------+-------------------------+---------------+
      | EGL_NONE                    | Marks the end of the    | N/A           |
      |                             | attribute-value list    |               |
      | EGL_IMAGE_PRESERVED_KHR     | Whether to preserve     | EGL_FALSE     |
      |                             | pixel data              |               |
      | EGL_PROTECTED_CONTENT_EXT   | Content protection      | EGL_FALSE     |
      |                             | state                   |               |
      +-----------------------------+-------------------------+---------------+
       Table bbb.  Legal attributes for eglCreateImageKHR <attrib_list>
       parameter

    Add the following paragraph to section 2.5.1 before "Errors" (describing
    eglCreateImageKHR):

        "If the value of attribute EGL_PROTECTED_CONTENT_EXT is EGL_TRUE
        and the EGLImage sources can be guaranteed to be protected, then the
        EGLImage is said to be protected. See section 2.7 (Protected Content)
        for more information about protected resources including EGLImages.

        If the value of attribute EGL_PROTECTED_CONTENT_EXT is EGL_FALSE then:

        - If EGLImage sources are not protected, the EGLImage is said to be
          not protected. See section 2.7 (Protected Content) for more
          information about non-protected resources including EGLImages.
        - If EGLImage sources are protected then the EGLImage content will
          be inaccessible to any client context irrespective of whether the
          context is protected or not. Trying to access such an EGLImage's
          content will result in undefined behavior."

   Add the following to the Errors list in section 2.5.1

        "If the value specified in <attrib_list> for EGL_PROTECTED_CONTENT_EXT
        is EGL_TRUE, and EGL and its client is unable to make guarantees
        regarding the protected state of the EGLImage source, the error
        EGL_BAD_ACCESS is generated."

Issues
    1) Can a protected context be shared with a non-protected context?

    RESOLVED - Yes. The rule that protected surfaces can only be used by
    protected contexts still applies. An example use case is where
    someone wants to render to unprotected textures within an unprotected
    context and then share it with a protected context to be used as a texture.

    2) Should all surfaces within a protected context be protected by default?

    RESOLVED - No, several implementations have limited amounts of protected
    memory, so the API will require opting into protected memory.

    3) Can these protected surfaces be used by stages other than fragment
    shader stage?

    RESOLVED - Some hardware can't handle this so this behavior is undefined
    unless there is explicit working in some new spec saying the behavior is
    defined.  This is put as an issue because this is an EGL extension and
    should not be controlling OpenGL functionality.

    4) Why is EGL_PROTECTED_CONTENT_EXT flag needed for EGLImages?

    RESOLVED - A few reasons for having an explicit flag instead
    of inferring the protected status from EGLImage sources -

      1) There are multiple EGL image extensions (EGL QCOM image, EGL
        android image and so on) that accept buffers from external modules
        instead of client resources or allow internally allocated memory.
        For these use cases a protected attribute is useful, so we want to
        keep this flag.
      2) An implementation might have a few non-standard setup steps that
        need to be completed before a protected EGL image can be accessed.
        This attribute along with a corresponding protected buffer will act
        as a signal for the graphics driver to initiate/complete any such
        steps.
      3) An application creating an image from an external resource may not
        be aware of the fact that the resource is protected or may be unable
        to access its content. The successful mapping of and access to a
        protected buffer through an EGLImage will be predicated on the
        buffer being protected, having a protected context and the intent of
        the application to access that buffer by passing in EGL_TRUE for the
        attribute EGL_PROTECTED_CONTENT_EXT.


Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    09/24/14   Ramesh   Initial draft.
     2    11/20/14   Rajeev   Second draft.
     3    03/07/16   mribble  Make EXT and clean up for release.
     4    03/10/16   mribble  Cleanup.
     5    03/18/16   mribble  Fix issues brought up by Khronos group.
     6    03/24/16   mribble  Resolved some small issues found by Jan-Harald.
     7    03/25/16   mribble  Fix createContext wording.
     8    03/30/16   mribble  Added issue 5.
     9    04/05/16   mribble  Added issue 6 and better defined eglImage case.
     10   04/08/16   rsmith   - Added general section on protected content.
                              Protected context, surface and image creation now
                              refer to the general protected content principles.
                              - Added explicit definition of which stages are
                              protected, including allowing for the protected
                              state of a stage to be undefined.
                              - Formalised interactions with
                              EGL_EXT_protected_surface.
                              - Removed references to the GPU protected mode,
                              including issue 3.
     11   04/10/16   mribble  Merge and cleanup.
     12   04/14/16  Jon Leech Cleanup formatting, reflow paragraphs and
                              quote additions consistently. Assign extension
                              number.
     13   12/06/21 Jeff Vigil Add queries for protected content attribute.
