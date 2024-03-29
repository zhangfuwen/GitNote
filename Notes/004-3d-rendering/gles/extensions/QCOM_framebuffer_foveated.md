# QCOM_framebuffer_foveated

Name

    QCOM_framebuffer_foveated

Name Strings

    GL_QCOM_framebuffer_foveated

Contributors

    Skyler Saleh
    Maurice Ribble
    Tate Hornbeck
    Jonathan Wicks
    Robert VanReenen

Contact

    Jeff Leger - jleger 'at' qti.qualcomm.com

Status

    Complete

Version

    Last Modified Date: May 10, 2017
    Revision: #11

Number

    OpenGL ES Extension #273

Dependencies

    OpenGL ES 2.0 is required.  This extension is written against OpenGL ES 3.2.

Overview

    Foveated rendering is a technique that aims to reduce fragment processing
    workload and bandwidth by reducing the average resolution of a framebuffer.
    Perceived image quality is kept high by leaving the focal point of
    rendering at full resolution.

    It exists in two major forms:

        - Static foveated(lens matched) rendering: where the gaze point is
        fixed with a large fovea region and designed to match up with the lens
        characteristics.
        - Eye-tracked foveated rendering: where the gaze point is continuously
        tracked by a sensor to allow a smaller fovea region (further reducing
        average resolution)

    Traditionally foveated rendering involves breaking a framebuffer's area
    into smaller regions such as bins, tiles, viewports, or layers which are
    rendered to individually. Each of these regions has the geometry projected
    or scaled differently so that the net resolution of these layers is less
    than the original framebuffer's resolution. When these regions are mapped
    back to the original framebuffer, they create a rendered result with
    decreased quality as pixels get further from the focal point.

    Foveated rendering is currently achieved by large modifications to an
    applications render pipelines to manually implement the required geometry
    amplifications, blits, and projection changes.  This presents a large
    implementation cost to an application developer and is generally
    inefficient as it can not make use of a platforms unique hardware features
    or optimized software paths. This extension aims to address these problems
    by exposing foveated rendering in an explicit and vendor neutral way, and by
    providing an interface with minimal changes to how an application specifies
    its framebuffer.

New Tokens

    Allowed in the config input in FramebufferFoveationConfigQCOM:

        FOVEATION_ENABLE_BIT_QCOM                    0x1
        FOVEATION_SCALED_BIN_METHOD_BIT_QCOM         0x2

New Procedures and Functions

    void FramebufferFoveationConfigQCOM(uint fbo,
                                       uint numLayers,
                                       uint focalPointsPerLayer,
                                       uint requestedFeatures,
                                       uint *providedFeatures);

    void FramebufferFoveationParametersQCOM(uint fbo,
                                           uint layer,
                                           uint focalPoint,
                                           float focalX,
                                           float focalY,
                                           float gainX,
                                           float gainY,
                                           float foveaArea);

Additions to Chapter 9 of the OpenGL ES 3.2 Specification

    The command

    void FramebufferFoveationConfigQCOM( uint fbo, uint numLayers,
    uint focalPointsPerLayer, uint requestedFeatures, uint *providedFeatures);

    is used to configure foveated rendering for the framebuffer object 'fbo'
    and to instruct the implementation to allocate any additional
    intermediate resources needed for foveated rendering.

    In order to enable foveation, this call must be issued prior to any
    operation which causes data to be written to a framebuffer attachment.
    Once this call is made for a framebuffer object, the fbo will remain a
    "foveated fbo". The following scenarios are unsupported conditions:

        1. Rendering to a non foveated fbo, then calling
           FramebufferFoveationConfigQCOM results in current framebuffer content
           becoming undefined.

        2. Rendering to a foveated fbo then switching attachments results in an
           error.

    Each layer of a foveated framebuffer, the max of which is specified by
    'numLayers', can have multiple focal points as controlled by
    'focalPointsPerLayer'. This enables applications that make use of double
    wide rendering to utilize foveated rendering and also allows more complex
    falloff characteristics to be modeled with multiple overlapping focal
    points. There are limitations to the number of focal points that an
    implementation can support with different enabled features or framebuffer
    formats. When an implementation can not support having as many focal points
    per layer as was specified in this function, it will fail with the error
    INVALID_VALUE. It is recommended that an application utilize as
    few focal points per layer as possible.

    The 'requestedFeatures' bitfield is used to specify which features an
    application would like to use.

    An explanation of each of the features is below:

        FOVEATION_ENABLE_BIT_QCOM: Is used to enable foveated rendering, if
        this bit is not specified foveated rendering will not be used.

        FOVEATION_SCALED_BIN_METHOD_BIT_QCOM: Requests that the implementation
        perform foveated rendering by dividing the framebuffer into a grid of
        subregions. Each subregions will be greater than or equal to one pixel
        and less than or equal to the full framebuffer. Then rendering the geometry
        to each of these regions with a different projection or scale. Then, finally
        upscaling the subregion to the native full resolution framebuffer.
        Quality in the scaled bin method is defined as a minimum pixel density
        which is the ratio of the resolution rendered compared to the native
        framebuffer.

    In the future it is expected that more features will be added, without
    breaking backwards compatibility.

    'providedFeatures' is a pointer to a uint that will be set to a new bitfield
    that tells the application which features the implementation provided for the
    current foveation configuration, in the same format as used in the 'requested
    Features' bitfield. This may include more or fewer features than the application
    requested.  The FOVEATION_ENABLE_BIT_QCOM will not be included if the
    implementation has to fallback to non-foveated rendering.

    If an application tries to make use of a feature that is not included in the
    'providedFeatures' bitfield, the results of the operation are implementation
    defined, but should not yield application termination.

    If this command is called with 'requestedFeatures' equal to zero, then the value
    of 'providedFeatures' will have FOVEATION_ENABLE_BIT_QCOM unset, and the other
    bits will be set or unset to indicate which foveation features are supported
    by the implementation.

    The command

    void FramebufferFoveationParametersQCOM(uint fbo,uint layer, uint focalPoint,
    float focalX, float focalY, float gainX, float gainY, float foveaArea);

    is used to control the falloff of the foveated rendering of 'focalPoint'
    for layer  'layer' in the framebuffer object 'fbo'. Multiple focal points
    per layer are provided to enable foveated rendering when different regions
    of a framebuffer represent different views, such as with double wide
    rendering. Values of 0 to the framebuffers focalPointsPerLayer-1 are valid
    for the 'focalPoint' input and specify which focal point's data to update
    for the layer.

    'focalX' and 'focalY' is used to specify the x and y coordinate
    of the focal point of the foveated framebuffer in normalized device
    coordinates. 'gainX' and 'gainY' are used to control how quickly the
    quality falls off as you get further away from the focal point in each
    axis. The larger these values are the faster the quality degrades.
    'foveaArea' is used to control the minimum size of the fovea region, the
    area before the quality starts to fall off. These parameters should be
    modified to match the lens characteristics.

    For the scaled bin method, these parameters define the minimum pixel
    density allowed for a given focal point at the location (px,py) on a
    framebuffer layer in NDC as:

    min_pixel_density=0.;
    for(int i=0;i<focalPointsPerLayer;++i){
        focal_point_density = 1./max((focalX[i]-px)^2*gainX[i]^2+
                            (focalY[i]-py)^2*gainY[i]^2-foveaArea[i],1.);
        min_pixel_density=max(min_pixel_density,focal_point_density);
    }

    While this function is continuous, it is worth noting that an
    implementation is allowed to decimate to a fixed number of supported
    quality levels, and it is allowed to group pixels into larger regions of
    constant quality level, as long as the implementation at least provides
    the quality level given in the above equation.

    Future supported foveation methods could have different definitions of
    quality.

    The default values for each of the focal points in a layer is:

    focalX=focalY=0;
    gainX=gainY=0;
    foveaArea=0;

    Which requires the entire framebuffer to be rendered at full quality.

    By specifying these constraints an application can fully constrain its
    render quality while leaving the implementation enough flexibility to
    render efficiently.

Errors

    OUT_OF_MEMORY is generated by FramebufferFoveationConfigQCOM if an
    implementation runs out of memory when trying to reserve the needed
    additional resources for the foveated framebuffer.

    INVALID_VALUE is generated by FramebufferFoveationConfigQCOM if 'fbo' is
    not a valid framebuffer.

    INVALID_VALUE is generated by FramebufferFoveationConfigQCOM if 'numLayers'
    is greater than GL_MAX_ARRAY_TEXTURE_LAYERS - 1.

    INVALID_VALUE is generated by FramebufferFoveationConfigQCOM if
    'numFocalPoints' is greater than implementation can support.

    INVALID_OPERATION is generated by FramebufferFoveationConfigQCOM if it is
    called for a fbo that has already been cofigured for foveated rendering.

    INVALID_VALUE is generated by FramebufferFoveationParametersQCOM if 'fbo'
    is not a valid framebuffer.

    INVALID_OPERATION is generated by FramebufferFoveationParametersQCOM if
    'fbo' has not been configured for foveated rendering.

    INVALID_VALUE is generated by FramebufferFoveationParametersQCOM if
    'layer' is greater than or equal to the numLayers that the fbo was
    previously configured for in FramebufferFoveationConfigQCOM.

    INVALID_VALUE is generated by FramebufferFoveationParametersQCOM if
    'numFocalPoints' is greater than implementation can support.

    INVALID_OPERATION is generated by any API call which causes a framebuffer
    attachment to be written to if the framebuffer attachments have changed for
    a foveated fbo.

    INVALID_OPERATION is generated if a rendering command is issued and the
    current bound program uses tessellation or geometry shaders.

Issues

    1. Are layered framebuffers supported?

    Layered framebuffers are supported to enable stereoscopic foveated
    rendering which is a main use case of this extension. When a layered
    framebuffer is used each layer has its own set of foveation parameters.

    2. How is foveation performed?

    How foveated rendering is performed to the framebuffer is implementation
    defined. However, if 'FOVEATION_SCALED_BIN_METHOD_BIT_QCOM' is set the
    implementation must perform foveation by dividing the framebuffer into a
    grid of subregions. Then rendering the geometry to each of these regions
    with a different projection or scale. And finally upscaling the subregion
    to the native full resolution framebuffer.

    When that bit is not set the implementation can use any algorithm it
    wants for foveated rendering as long as it meets the application
    requested features.

    3. How are MRTs handled?

    Every framebuffer attachment uses the same quality in a given region.

    4. How does gl_FragCoord work?

    When using the scaled bin method, gl_FragCoord will be scaled to match
    the relative location in a foveated framebuffer. This means the absolute
    value of gl_FragCoord will not be correct in lower resolution areas,
    but the value relative to the full resolution will be consistent.

    5. How is depth handled?

    Depth surfaces can be used during foveated rendering, but the contents
    of the depth buffer will not be resolved out. The implementation can
    do an implicit discard of the depth buffer. The reasoning here is that
    the upsample of depth during resolve would produce irrelevant results.

    6. How does unresolving from a foveated framebuffer work?

    Loading from a foveated framebuffer is undefined, so the app must be
    sure not to trigger mid frame flushes. In the dynamic foveation case
    the focal point can move constantly. If a region A of a frame 0 was
    rendered at a lower quality because the focal point was far away,
    then the focal point moved to cover region A during frame 1, the
    unresolve could not reconstruct the full quality region A. The
    app must be careful to fully clear the surface and remove mid frame
    flushes to prevent unresolves.

Examples:

    (1) Initialize a foveated framebuffer

        // Allocate and initialize a regular framebuffer and attachments
        GLuint fbo = createFramebufferAndAttachments();
        GLuint providedFeatures;
        glFramebufferFoveationConfigQCOM(fbo,1,1, GL_FOVEATION_ENABLE_BIT_QCOM, &providedFeatures);
        if(!(providedFeatures & GL_FOVEATION_ENABLE_BIT_QCOM)) {
            // Failed to enable foveation
        }

    (2) Setup static foveated rendering

        // Insert code from #1
        GLfloat focalX=0.f, focalY=0.f;  // Setup focal point at the center of screen
        GLfloat gainX=4.f, gainY=4.f;  // Increase these for stronger foveation
        glFramebufferFoveationParametersQCOM(fbo, 0, 0, focalX, focalY, gainX, gainY, 0.f);

    (3) Change eye position for eye tracked foveated rendering

        // Code called whenever the eye position changes
        // It is best to position this call both before rendering anything to
        //   a fbo and right before Flush or changing FBO since some
        //   some implementations can apply this state late by patching command
        //   buffers.
        glFramebufferFoveationParametersQCOM(fbo, 0, 0, focalX, focalY, gainX, gainY, 0.f);

    (4) Setting parameters for a multiview stereo framebuffer

        //focalPointsPerLayer should be 1
        float focalX1=0.f,focalY1=0.f;  // Gaze of left eye
        float focalX2=0.f,focalY2=0.f;  // Gaze of right eye
        float gain_x=10.f,gain_y=10.f;  // Strong foveation
        glFramebufferFoveationParametersQCOM(fbo, 0, 0, focalX1, focalY1,gainX, gainY, 0.f);
        glFramebufferFoveationParametersQCOM(fbo, 1, 0, focalX2, focalY2,gainX, gainY, 0.f);

    (5) Setting parameters for a double wide stereo framebuffer

        //focalPointsPerLayer should be 2
        float focalX1=0.f,focalY1=0.f;  // Gaze of left eye
        float focalX2=0.f,focalY2=0.f;  // Gaze of right eye
        float gainX=10.f,gainY=10.f;
        glFramebufferFoveationParametersQCOM(fbo, 0, 0, focalX1*0.5f-0.5f, focalY1, gainX*2.f ,gainY, 0.f);
        glFramebufferFoveationParametersQCOM(fbo, 0, 1, focalX2*0.5f+0.5f, focalY2, gainX*2.f ,gainY, 0.f);


Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    05/19/16  ssaleh    Initial draft.
     2    05/27/16  ssaleh    Made the extension much more explicit.
     3    07/08/16  ssaleh    Further refinements.
     4    08/11/16  ssaleh    Specified bitfield values
     5    08/19/16  ssaleh    Added support for double wide rendering
     6    08/24/16  mribble   Name changes and cleanup
     7    08/24/16  ssaleh    Add examples
     8    10/14/16  tateh     Clarify gl_FragCoord
     9    01/04/17  tateh     Update entry points and cleanup
    10    02/17/17  jleger    Convert from EXT to QCOM extension.
    11    05/10/17  tateh     Minor cleanup
