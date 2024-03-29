# OES_surfaceless_context

Name

    OES_surfaceless_context

Name Strings

    GL_OES_surfaceless_context

Contributors

    Kristian Hoegsberg, Intel
    Steven Holte, NVIDIA
    Greg Roth, NVIDIA

Contact

    Steven Holte, NVIDIA (sholte 'at' nvidia.com)

Notice

    Copyright (c) 2010-2013 The Khronos Group Inc. Copyright terms at
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

    Complete.
    Approved by the OpenGL ES Working Group.
    Ratified by the Khronos Board of Promoters on July 28, 2012.

Version

    Version 4, 2012/05/30

Number

    OpenGL ES Extension #116

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification
    but can apply to OpenGL ES 1.1 with the GL_OES_framebuffer_object
    extension.

    Support for creating contexts that this extension applies to may
    require extensions to companion APIs (see
    EGL_KHR_surfaceless_context)

Overview

    Applications that only want to render to framebuffer objects should
    not need to create a throw-away EGL surface (typically a 1x1
    pbuffer) just to get a current context.  The EGL extension
    KHR_surfaceless_context provides a mechanism for making a context
    current without a surface.  This extensions specifies the behaviour
    of OpenGL ES 1.x and OpenGL ES 2.0 when such a context is made
    current.

New Procedures and Functions

    None

New Tokens

    Returned by glCheckFramebufferStatusOES and glCheckFramebufferStatus:

        GL_FRAMEBUFFER_UNDEFINED_OES                    0x8219

Additions to Chapter 2 'OpenGL ES Operation' of the OpenGL ES 2.0
Specification:

    In section 2.1 'OpenGL ES Fundamentals', replace the paragraphs
    beginning:

    "   The GL interacts with two classes of framebuffers: window-
    system-provided framebuffers and application-created framebuffers ...
        The effects of GL commands on the window-system-provided
    framebuffer are ultimately controlled by the window-system that
    allocates framebuffer resources ...
        The initialization of a GL context itself occurs when the
    window-system allocates a window for GL rendering and is influenced
    by the state of the windowsystem-provided framebuffer"

    with the following paragraphs:

    "    The GL interacts with two classes of framebuffers: window
    system-provided and application-created. There is at most one window
    system-provided framebuffer at any time, referred to as the default
    framebuffer. Application-created framebuffers, referred to as
    framebuffer objects, may be created as desired. These two types of
    framebuffer are distinguished primarily by the interface for
    configuring and managing their state.
        The effects of GL commands on the default framebuffer are
    ultimately controlled by the window system, which allocates
    framebuffer resources, determines which portions of the default
    framebuffer the GL may access at any given time, and communicates to
    the GL how those portions are structured. Therefore, there are no GL
    commands to initialize a GL context or configure the default
    framebuffer.
        Similarly, display of framebuffer contents on a physical display
    device (including the transformation of individual framebuffer
    values by such techniques as gamma correction) is not addressed by
    the GL.
        Allocation and configuration of the default framebuffer occurs
    outside of the GL in conjunction with the window system, using
    companion APIs, such as EGL. Allocation and initialization of GL
    contexts is also done using these companion APIs. GL contexts can
    typically be associated with different default framebuffers, and
    some context state is determined at the time this association is
    performed.
        It is possible to use a GL context without a default framebuffer,
    in which case a framebuffer object must be used to perform all
    rendering. This is useful for applications needing to perform
    offscreen rendering."

    In the last paragraph of section 2.12 'Controlling the viewport',
    after the sentence:

    " In the initial state, w and h are set to the width and height,
    respectively, of the window into which the GL is to do its
    rendering."

    Add the sentence:

    " If no default framebuffer is associated with the GL context (see
    chapter 4), then w and h are initially set to zero."

Additions to Chapter 4 'Per-Fragment Operations and the Framebuffer', of
the OpenGL ES 2.0 Specification:

    In the introduction, after the sentence:

    " Further, and implementation or context may not provide depth or
    stencil buffers."

    Add the sentence:
    " If no default framebuffer is associated with the GL context, the
    framebuffer is incomplete except when a framebuffer object is bound.
    (see sections 4.4.1 and 4.4.5)"

    In the last paragraph of section 4.1.2 'Scissor Test', after the
    sentence

    " The state required consists of four integer values and a bit
    indicating whether the test is enabled or disabled. In the initial
    state left = bottom = 0; width and height are determined by the size
    of the GL window."

    Add the sentence:

    " If the default framebuffer is bound but no default framebuffer is
    associated with the GL context (see chapter 4), then width and
    height are initially set to zero."

    In section 4.4.5 'Framebuffer Completeness', before the first
    paragraph, add the paragraphs:

    "   A framebuffer must be framebuffer complete to effectively be
    used as the draw or read framebuffer of the GL.
        The default framebuffer is always complete if it exists; however,
    if no default framebuffer exists (no window system-provided drawable
    is associated with the GL context), it is deemed to be incomplete."

    In the subsection 'Framebuffer Completeness', add to the list of
    rules for framebuffer completeness and associated errors:

    "* if target is the default framebuffer, the default framebuffer
    exists. { FRAMEBUFFER_UNDEFINED_OES } "

    To the list of actions which may affect framebuffer completeness,
    add the action:

    " Associating a different window system-provided drawable, or no
    drawable, with the default framebuffer using a window system binding
    API such as EGL."

Revision History

    Version 4, 2012/05/30 (Greg Roth) - OESify. Add suffix. Omit
    indiscression. Revise widths.

    Version 3, 2012/05/29 (Steven Holte) - Typo corrections.

    Version 2, 2012/04/13 (Steven Holte) - Language modifications
    expanded to include harmonize with related specifications.

    Version 1, 2010/08/19 (Kristian Hoegsberg) - Initial draft, based
    on Jon's wording in the EGL_KHR_surfaceless_gles extension.
