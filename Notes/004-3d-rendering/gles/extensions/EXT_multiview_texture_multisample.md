# EXT_multiview_texture_multisample

Name

    EXT_multiview_texture_multisample

Name Strings

    GL_EXT_multiview_texture_multisample

Contact

    Robert Menzel, NVIDIA Corporation (rmenzel 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA Corporation
    James Helferty, NVIDIA Corporation
    Kedarnath Thangudu, NVIDIA Corporation

Status

    Complete.

Version

    Last Modified Date:  May 13, 2019
    Author Revision: 1

Number

    OpenGL Extension #537
    OpenGL ES Extension #318

Dependencies

    OpenGL 4.0 or OpenGL ES 3.2 are required.

    This extension is written against the OpenGL 4.6 specification
    (Core Profile) (February 2, 2019) and OpenGL ES 3.2 specification
    (February 2, 2019).

    OVR_multiview is required.

Overview

    OVR_multiview introduced multiview rendering to OpenGL and OpenGL ES.
    
    This extension removes one of the limitations of the OVR_multiview 
    extension by allowing the use of multisample textures during multiview rendering.
    
    This is one of two extensions that allow multisampling when using 
    OVR_multiview. Each supports one of the two different approaches to 
    multisampling in OpenGL and OpenGL ES:
    
        Core OpenGL and OpenGL ES 3.1+ have explicit support for multisample 
        texture types, such as TEXTURE_2D_MULTISAMPLE. Applications can access 
        the values of individual samples and can explicitly "resolve" the 
        samples of each pixel down to a single color.
        
        The extension EXT_multisampled_render_to_texture provides support for 
        multisampled rendering to non-multisample texture types, such as 
        TEXTURE_2D. The individual samples for each pixel are maintained 
        internally by the implementation and can not be accessed directly 
        by applications. These samples are eventually resolved implicitly to 
        a single color for each pixel.
        
    This extension supports the first multisampling style with multiview 
    rendering; the OVR_multiview_multisampled_render_to_texture extension 
    supports the second style. Note that support for one of these multiview 
    extensions does not imply support for the other.
    
    
New Tokens

    None.

    
New Procedures and Functions

    None.
    
    
Modifications to Chapter 9 of the OpenGL ES 3.2 Specification as well as
Chapter 9 of the OpenGL 4.6 Specification (Framebuffers and Framebuffer 
Objects) 
    
Modifications to all sections added and/or modified by OVR_multiview

    Where OVR_multiview references a "two-dimensional array texture", replace 
    this with "two-dimensional array texture or two-dimensional multisample 
    array texture" to explicitly allow rendering to multisampled textures.
    
    The following is an explicit list of these changes:
    
    In subsection 9.2.2.2 (introduced by OVR_multiview) replace
        "Finally, multiple layers of two-dimensional array textures can be
        attached to an attachment point."
    with
        "Finally, multiple layers of two-dimensional array textures or 
        two-dimensional multisample array textures can be
        attached to an attachment point."
    
    In the additions to section 9.2.3 (Framebuffer Object Queries) replace
        "If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR and the value
        of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a two-dimensional array texture..."
    with
        "If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR and the value
        of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a two-dimensional array texture
        or a two-dimensional multisample array texture..."
    
    Also replace
        "If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR and the
        value of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a two-dimensional array
        texture..."
    with
        "If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR and the
        value of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a two-dimensional array
        texture or a two-dimensional multisample array texture..."
    
    In the addition to the end of section 9.2.8 (Attaching Textures to a Framebuffer)
    replace
        "Multiple layers of a two-dimensional array texture can be
        attached as one of the logical buffers of a framebuffer object with the
        commands"
    with
        "Multiple layers of a two-dimensional array texture or a two-dimensional 
        multisample array texture can be
        attached as one of the logical buffers of a framebuffer object with the
        commands"
    
    Later in that section under "Errors" replace
        "An INVALID_VALUE error is generated if:
        - <texture> is a two-dimensional array texture and <baseViewIndex> +
          <numViews> is larger than the value of MAX_ARRAY_TEXTURE_LAYERS."
    with
        "An INVALID_VALUE error is generated if:
        - <texture> is a two-dimensional array texture or a two-dimensional 
        multisample array texture and <baseViewIndex> +
          <numViews> is larger than the value of MAX_ARRAY_TEXTURE_LAYERS."
    
    Also replace
        "An INVALID_OPERATION error is generated if texture is non-zero and is not
        the name of a two-dimensional array texture."
    with
        "An INVALID_OPERATION error is generated if texture is non-zero and is not
        the name of a two-dimensional array texture or a two-dimensional 
        multisample array texture."
      
    
Issues

    (1) This extension is based on an OVR extension, why call it EXT?
    
    While started as a single vendor extension, OVR_multiview and OVR_multiview2
    are already supported by multiple vendors. This new extension also has the 
    support from multiple vendors to be specified as EXT.
    
    (2) Can this extension be used together with 
    OVR_multiview_multisampled_render_to_texture if both are supported?
    
    The two extensions can be used together in the same application, but not in 
    the same framebuffer. OVR_multiview_multisampled_render_to_texture is based 
    on EXT_multisampled_render_to_texture, which explicitly does not permit a 
    single framebuffer to contain explicit multisample texture types (from 
    OpenGL ES 3.1 and core OpenGL) and "multisampled" bindings of non-multisample 
    texture types.
    
Revision History

      Rev.    Date    Author    Changes
      ----  --------  --------  -----------------------------------------
        1   05/13/19  rmenzel   Initial version.
