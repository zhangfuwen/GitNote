# QCOM_texture_foveated

Name

    QCOM_texture_foveated

Name Strings

    GL_QCOM_texture_foveated

Contributors

    Tate Hornbeck
    Jonathan Wicks
    Robert VanReenen
    Matthew Hoffman
    Jeff Leger

Contact

    Jeff Leger - jleger 'at' qti.qualcomm.com

Status

    Complete

Version

    Last Modified Date: Oct 06, 2017
    Revision: #5

Number

     OpenGL ES Extension #293

Dependencies

    OpenGL ES 2.0 is required.  This extension is written against OpenGL ES 3.2.

    This interacts with QCOM_framebuffer_foveated.

Overview

    Foveated rendering is a technique that aims to reduce fragment processing
    workload and bandwidth by reducing the average resolution of a render target.
    Perceived image quality is kept high by leaving the focal point of
    rendering at full resolution.

    It exists in two major forms:

        - Static foveated (lens matched) rendering: where the gaze point is
        fixed with a large fovea region and designed to match up with the lens
        characteristics.
        - Eye-tracked foveated rendering: where the gaze point is continuously
        tracked by a sensor to allow a smaller fovea region (further reducing
        average resolution)

    Traditionally foveated rendering involves breaking a render target's area
    into smaller regions such as bins, tiles, viewports, or layers which are
    rendered to individually. Each of these regions has the geometry projected
    or scaled differently so that the net resolution of these layers is less
    than the original render target's resolution. When these regions are mapped
    back to the original render target, they create a rendered result with
    decreased quality as pixels get further from the focal point.

    Foveated rendering is currently achieved by large modifications to an
    applications render pipelines to manually implement the required geometry
    amplifications, blits, and projection changes.  This presents a large
    implementation cost to an application developer and is generally
    inefficient as it can not make use of a platforms unique hardware features
    or optimized software paths. This extension aims to address these problems
    by exposing foveated rendering in an explicit and vendor neutral way, and by
    providing an interface with minimal changes to how an application specifies
    its render targets.

New Tokens

    Accepted as a value for <pname> for the TexParameter{if} and
    TexParameter{if}v commands and for the <pname> parameter of
    GetTexParameter{if}v:

        TEXTURE_FOVEATED_FEATURE_BITS_QCOM           0x8BFB
        TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM      0x8BFC

    Accepted as the <pname> parameter of GetTexParameter{if}v:

        TEXTURE_FOVEATED_FEATURE_QUERY_QCOM          0x8BFD
        TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM 0x8BFE

    Accepted as a value to <param> for the TexParameter{if} and
    to <params> for the TexParameter{if}v commands with a <pname> of
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM; returned as possible values for
    <params> when GetTexParameter{if}v is queried with a <pname> of
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM:

        FOVEATION_ENABLE_BIT_QCOM                    0x1
        FOVEATION_SCALED_BIN_METHOD_BIT_QCOM         0x2

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_FOVEATION_QCOM        0x8BFF

Add new rows to Table 8.19 (Texture parameters and their values):

    Name                               | Type | Legal Values
    ------------------------------------------------------------
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM | uint | 0,
                                                FOVEATION_ENABLE_BIT_QCOM,
                                                FOVEATION_ENABLE_BIT_QCOM |
                                                FOVEATION_SCALED_BIN_METHOD_BIT_QCOM)

    TEXTURE_FOVEATED_FEATURE_QUERY_QCOM | uint | 0,
                                                 FOVEATION_ENABLE_BIT_QCOM,
                                                 FOVEATION_ENABLE_BIT_QCOM |
                                                 FOVEATION_SCALED_BIN_METHOD_BIT_QCOM)

    TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM | uint | Any integer greater than 0

    TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM | float | Any float between 0.0 and 1.0

Add new rows to Table 21.10 Textures (state per texture object)

    Get value | Type | Get Command | Initial Value | Description | Sec
    ------------------------------------------------------------------
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM | Z+ | GetTexParameter{if}v | 0 | Foveation State | 8.19

    TEXTURE_FOVEATED_FEATURE_QUERY_QCOM | Z+ | GetTexParameter{if}v | see sec 8.19 | Supported foveation state | 8.19

    TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM | Z+ | GetTexParameter{if}v | see sec 8.19 | Number of supported focal points per texture layer | 8.19

    TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM | R[0.0,1.0] | GetTexParameter{if}v | 0.0 | Minimum pixel density allowed | 8.19

New Procedures and Functions

    void TextureFoveationParametersQCOM(uint  texture,
                                        uint  layer,
                                        uint  focalPoint,
                                        float focalX,
                                        float focalY,
                                        float gainX,
                                        float gainY,
                                        float foveaArea);

Additions to the end of section 8.19 of the OpenGL ES 3.2 Specification

    TEXTURE_FOVEATED_FEATURE_QUERY_QCOM is a texture property that can only be
    queried via GetTexParameter{if}v. This will return the implementation's
    supported foveation methods.

    glGetTexParameteriv(GL_TEXTURE_2D,
                        GL_TEXTURE_FOVEATED_FEATURE_QUERY_QCOM,
                        &query);

    if ((query & FOVEATION_ENABLE_BIT_QCOM == FOVEATION_ENABLE_BIT_QCOM) &&
        (query & FOVEATION_SCALED_BIN_METHOD_BIT_QCOM ==
                                   FOVEATION_SCALED_BIN_METHOD_BIT_QCOM))
    {
         // Implementation supports scaled bin method of foveation
    }

    TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM is a texture property that
    can only be queried GetTexParameter{if}v. This will return the number of
    focal points per texture layer that the implementation supports. This must
    be greater than 0.

    TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM defines the minimum pixel density
    that can be used for scaled bin foveated rendering.

    TEXTURE_FOVEATED_FEATURE_BITS_QCOM can be used to enable foveation for a
    texture render target.

    An explanation of each of the features is below:

        FOVEATION_ENABLE_BIT_QCOM: Is used to enable foveated rendering, if
        this bit is not specified foveated rendering will not be used.

        FOVEATION_SCALED_BIN_METHOD_BIT_QCOM: Requests that the implementation
        perform foveated rendering by dividing the texture render target into a
        grid of subregions. Each subregions will be greater than or equal to one pixel
        and less than or equal to the full size of the texture. Then rendering
        the geometry to each of these regions with a different projection or scale.
        Then, finally upscaling the subregion to the native texture resolution.
        Quality in the scaled bin method is defined as a minimum pixel density
        which is the ratio of the resolution rendered compared to the native
        texture.

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_FOVEATED_FEATURE_BITS_QCOM,
                    GL_FOVEATION_ENABLE_BIT_QCOM |
                    GL_FOVEATION_SCALED_BIN_METHOD_BIT_QCOM);

    This will set a texture as foveated so all subsequent rendering to
    this texture will be foveated as long as the FOVEATION_ENABLE_BIT_QCOM
    is set. Foveation is a texture property that is only applicable for
    rendering operations, it does not affect traditional texture functions
    like TexImage2D or TexSubImage2D.

    The command

    void TextureFoveationParametersQCOM(uint  texture,
                                        uint  layer,
                                        uint  focalPoint,
                                        float focalX,
                                        float focalY,
                                        float gainX,
                                        float gainY,
                                        float foveaArea);

    is used to control the falloff of the foveated rendering of 'focalPoint'
    for layer  'layer' in the texture object 'texture'. Multiple focal points
    per layer are provided to enable foveated rendering when different regions
    of a texture represent different views, such as with double wide
    rendering. TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM focal points are
    supported per texture so values of 0 to
    TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM minus one are valid for the
    'focalPoint' input and specify which focal point's data to update for the
    layer.

    'focalX' and 'focalY' is used to specify the x and y coordinate
    of the focal point of the foveated texture in normalized device
    coordinates. 'gainX' and 'gainY' are used to control how quickly the
    quality falls off as you get further away from the focal point in each
    axis. The larger these values are the faster the quality degrades.
    'foveaArea' is used to control the minimum size of the fovea region, the
    area before the quality starts to fall off. These parameters should be
    modified to match the lens characteristics.

    For the scaled bin method, these parameters define the minimum pixel
    density allowed for a given focal point at the location (px,py) on a
    texture layer in NDC as:

    min_pixel_density=0.;
    for(int i=0;i<focalPointsPerLayer;++i){
        focal_point_density = 1./max((focalX[i]-px)^2*gainX[i]^2+
                            (focalY[i]-py)^2*gainY[i]^2-foveaArea[i],1.);
        min_pixel_density=max(min_pixel_density,focal_point_density);
        min_pixel_desnsity=max(min_pixel_desnsity,
                               TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM);
    }

    While this function is continuous, it is worth noting that an
    implementation is allowed to decimate to a fixed number of supported
    quality levels, and it is allowed to group pixels into larger regions of
    constant quality level, as long as the implementation at least provides
    the quality level given in the above equation.

    Future supported foveation methods could have different definitions of
    quality.

    The default values for each of the focal points in a layer is:

    focalX = focalY = 0.0;
    gainX  = gainY  = 0.0;
    foveaArea = 0.0;

    Which requires the entire render target to be rendered at full quality.

    By specifying these constraints an application can fully constrain its
    render quality while leaving the implementation enough flexibility to
    render efficiently.

Additions to Chapter 9.4 (Framebuffer Completeness) of the OpenGL ES 3.2 Specification

    More than one color attachment is foveated.

    { FRAMEBUFFER_INCOMPLETE_FOVEATION_QCOM }

    Depth or stencil attachments are foveated textures.

    { FRAMEBUFFER_INCOMPLETE_FOVEATION_QCOM }

    The framebuffer has been configured for foveation via QCOM_framebuffer_foveated
    and any color attachment is a foveated texture.

    { FRAMEBUFFER_INCOMPLETE_FOVEATION_QCOM }

Errors

    INVALID_VALUE is generated by TextureFoveationParametersQCOM if 'texture'
    is not a valid texture object.

    INVALID_OPERATION is generated by TextureFoveationParametersQCOM if
    'texture' has not been set as foveated. i.e. 'texture's parameter
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM does not contain
    FOVEATION_ENABLE_BIT_QCOM.

    INVALID_VALUE is generated by TextureFoveationParametersQCOM if
    'focalPoint' is larger than TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM
    minus one.

    INVALID_ENUM is generated by TexParameter{if} or TexParamter{if}v
    if <pname> is TEXTURE_FOVEATED_FEATURE_QUERY_QCOM or
    TEXTURE_FOVEATED_NUM_FOCAL_POINTS_QUERY_QCOM.

    INVALID_ENUM is generated by TexParameter{if} or TexParamter{if}v
    if <pname> is TEXTURE_FOVEATED_FEATURE_BITS_QCOM and <param> has
    other bits set besides the legal values listed in table 8.19.

    INVALID_OPERATION is generated by TexParameter{if} or TexParamter{if}v
    if <pname> is TEXTURE_FOVEATED_FEATURE_BITS_QCOM and <param> does not have
    FOVEATION_ENABLE_BIT_QCOM bit set, but the texture's parameter
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM already contains FOVEATION_ENABLE_BIT_QCOM.
    i.e. Once foveation has been enabled for a texture, it cannot be disabled.

    INVALID_OPERATION is generated by TexParameter{if} or TexParamter{if}v
    if <pname> is TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM and <param> is a
    float less than 0.0 or greater than 1.0.

    INVALID_OPERATION is generated by TexParameter{if} or TexParamter{if}v if
    <pname> is GL_TEXTURE_FOVEATED_FEATURE_BITS_QCOM and <param> contains
    FOVEATION_ENABLE_BIT_QCOM, but the query of TEXTURE_FOVEATED_FEATURE_QUERY_QCOM
    of <target> does not contain FOVEATION_ENABLE_BIT_QCOM.

Issues

    1. Are texture arrays supported?

    Texture arrays are supported to enable stereoscopic foveated
    rendering which is a main use case of this extension. When a texture
    array is used as a foveated render target, each slice has its own set
    of foveation parameters.

    2. How is foveation performed?

    How foveated rendering is performed to the texture render target is implementation
    defined. However, if 'FOVEATION_SCALED_BIN_METHOD_BIT_QCOM' is set the
    implementation must perform foveation by dividing the render target into a
    grid of subregions. Then rendering the geometry to each of these regions
    with a different projection or scale. And finally upscaling the subregion
    to the native full resolution render target.

    When that bit is not set the implementation can use any algorithm it
    wants for foveated rendering as long as it meets the application
    requested features.

    3. How are MRTs handled?

    Only one color attachment may be foveated, all other color attachments
    will inherit the foveated color attachment's foveation state.

    4. Effect on screenspace shader built-ins

    When using the scaled bin method, a number of screenspace built-ins
    can produce unexpected results.

    gl_FragCoord will be scaled to match the relative location in a
    foveated texture. This means the absolute value of gl_FragCoord
    will not be correct in lower resolution areas, but the value relative
    to the full resolution will be consistent.

    interpolateAtOffset, gl_PointSize, gl_SamplePosition, dFdx, dFdy,
    glLineWidth will have no corrective scaling applied and thus could
    have unexpected results.

    5. How is depth/stencil handled?

    Foveation cannot be enabled for depth or stencil texture attachments.  However,
    they will inherit foveation from a foveated color attachment attached to the
    same framebuffer. In this case the depth and/or stencil attachments should
    be discarded or invalidated after rendering, as the upscaled depth contents
    are unlikely to be useful and may cause undesired rendering artifacts when used.

    6. Does foveation have any effect on BlitFramebuffer.

    No, there is no option to foveate a BlitFramebuffer blit. You can BlitFramebuffer
    from a fbo with a foveated color attachment.

    7. Rendering to a foveated texture multiple times per flush

    The application must be careful to fully clear, or discard/invalidate, any
    foveated attachments before rendering.  Failure to do so would cause
    unresolves of foveated content which may be undesirable (e.g. cases where
    the foveation parameters or focal point has changed between resolves).
    To prevent this the implementation may disable foveation for any rendering
    to a foveated attachment that requires unresolves.  Texture state related
    to foveation, like TEXTURE_FOVEATED_FEATURE_BITS_QCOM and
    TEXTURE_FOVEATED_MIN_PIXEL_DENSITY_QCOM will not be affected.

    8. Interactions with QCOM_framebuffer_foveated

    It is illegal to mix usage of these extensions. If a framebuffer has been
    configured for foveation via QCOM_framebuffer_foveated, no attached textures
    can be configured for foveation via QCOM_texture_foveated. The framebuffer
    will be incomplete for this situation.

    9. Implementation forced down non-tiled path

    Certain feature sets may force an implementation to perform non tiled rendering.
    The implemenation may implicitly disable foveation for these cases. Some
    potential examples include tessellation, geometry shaders, or compute.

Examples:

    (1) Setup a foveated texture

        // Allocate a texture
        GLuint foveatedTexture;
        glGenTextures(1, &foveatedTexture);
        glBindTexture(GL_TEXTURE_2D, foveatedTexture);
        ...
        // Set texture as foveated
        glTexParameteri(GL_TEXTURE_2D,
                        GL_TEXTURE_FOVEATED_FEATURE_BITS_QCOM,
                        GL_FOVEATION_ENABLE_BIT_QCOM | GL_FOVEATION_SCALED_BIN_METHOD_BIT_QCOM);
        ...
        // Rendering to foveatedTexture
        ...
        // Set foveation parameters on texture
        glTextureFoveationParametersQCOM(foveatedTexture, 0, 0, focalX, focalY, gainX, gainY, 0.0f);
        glFlush();

    (2) Setting parameters for a multiview stereo texture array

        float focalX1 = 0.0f; // Gaze of left eye
        float focalY1 = 0.0f; // Gaze of left eye
        float focalX2 = 0.0f; // Gaze of right eye
        float focalY2 = 0.0f; // Gaze of right eye
        float gainX   = 4.0f; // Weak foveation
        float gainY   = 4.0f; // Weak foveation

        glTextureFoveationParametersQCOM(foveatedTextureArray,
                                         0,
                                         0,
                                         focalX1,
                                         focalY1,
                                         gainX,
                                         gainY,
                                         2.0f);
        glTextureFoveationParametersQCOM(foveatedTextureArray,
                                         1,
                                         0,
                                         focalX2,
                                         focalY2,
                                         gainX,
                                         gainY,
                                         2.0f);

    (3) Setting parameters for a double wide stereo texture

        float focalX1 = -0.5f; // Gaze of left eye
        float focalY1 =  0.0f; // Gaze of left eye
        float focalX2 =  0.5f; // Gaze of right eye
        float focalY2 =  0.0f; // Gaze of right eye
        float gainX   =  8.0f; // Strong foveation
        float gainY   =  8.0f; // Strong foveation

        glTextureFoveationParametersQCOM(foveatedTexture,
                                         0,
                                         0,
                                         focalX1,
                                         focalY1,
                                         gainX * 2.0f,
                                         gainY,
                                         8.0f);
        glTextureFoveationParametersQCOM(foveatedTexture,
                                         0,
                                         1,
                                         focalX2,
                                         focalY2,
                                         gainX * 2.0f,
                                         gainY,
                                         8.0f);

Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    08/22/17   tateh     Initial spec
     2    09/20/17   tateh     Clarified screenspace shader issues. Added
                               way to query number of supported focal points.
     3    09/25/17   tateh     Add max scale factor texture parameter.
     4    10/3/17    tateh     Add Table 21.10 modifications and issue #9
     5    10/6/17    tateh     Changed max scale factor to min pixel density
     6    01/8/17    tateh     Removed depth/stencil discard and invalidate
                               language.
