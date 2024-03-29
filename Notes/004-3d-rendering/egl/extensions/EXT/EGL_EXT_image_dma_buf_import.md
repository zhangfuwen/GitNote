# EXT_image_dma_buf_import

Name

    EXT_image_dma_buf_import

Name Strings

    EGL_EXT_image_dma_buf_import

Contributors

    Jesse Barker
    Rob Clark
    Tom Cooksey

Contacts

    Jesse Barker (jesse 'dot' barker 'at' linaro 'dot' org)
    Tom Cooksey (tom 'dot' cooksey 'at' arm 'dot' com)

Status

    Complete.

Version

    Version 7, December 13, 2013

Number

    EGL Extension #53

Dependencies

    EGL 1.2 is required.

    EGL_KHR_image_base is required.

    The EGL implementation must be running on a Linux kernel supporting the
    dma_buf buffer sharing mechanism.

    This extension is written against the wording of the EGL 1.2 Specification.

Overview

    This extension allows creating an EGLImage from a Linux dma_buf file
    descriptor or multiple file descriptors in the case of multi-plane YUV
    images.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <target> parameter of eglCreateImageKHR:

        EGL_LINUX_DMA_BUF_EXT          0x3270

    Accepted as an attribute in the <attrib_list> parameter of
    eglCreateImageKHR:

        EGL_LINUX_DRM_FOURCC_EXT        0x3271
        EGL_DMA_BUF_PLANE0_FD_EXT       0x3272
        EGL_DMA_BUF_PLANE0_OFFSET_EXT   0x3273
        EGL_DMA_BUF_PLANE0_PITCH_EXT    0x3274
        EGL_DMA_BUF_PLANE1_FD_EXT       0x3275
        EGL_DMA_BUF_PLANE1_OFFSET_EXT   0x3276
        EGL_DMA_BUF_PLANE1_PITCH_EXT    0x3277
        EGL_DMA_BUF_PLANE2_FD_EXT       0x3278
        EGL_DMA_BUF_PLANE2_OFFSET_EXT   0x3279
        EGL_DMA_BUF_PLANE2_PITCH_EXT    0x327A
        EGL_YUV_COLOR_SPACE_HINT_EXT    0x327B
        EGL_SAMPLE_RANGE_HINT_EXT       0x327C
        EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT  0x327D
        EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT    0x327E

    Accepted as the value for the EGL_YUV_COLOR_SPACE_HINT_EXT attribute:

        EGL_ITU_REC601_EXT   0x327F
        EGL_ITU_REC709_EXT   0x3280
        EGL_ITU_REC2020_EXT  0x3281

    Accepted as the value for the EGL_SAMPLE_RANGE_HINT_EXT attribute:

        EGL_YUV_FULL_RANGE_EXT    0x3282
        EGL_YUV_NARROW_RANGE_EXT  0x3283

    Accepted as the value for the EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT &
    EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT attributes:

        EGL_YUV_CHROMA_SITING_0_EXT    0x3284
        EGL_YUV_CHROMA_SITING_0_5_EXT  0x3285


Additions to Chapter 2 of the EGL 1.2 Specification (EGL Operation)

    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

   "Values accepted for <target> are listed in Table aaa, below.

      +-------------------------+--------------------------------------------+
      |  <target>               |  Notes                                     |
      +-------------------------+--------------------------------------------+
      |  EGL_LINUX_DMA_BUF_EXT  |   Used for EGLImages imported from Linux   |
      |                         |   dma_buf file descriptors                 |
      +-------------------------+--------------------------------------------+
       Table aaa.  Legal values for eglCreateImageKHR <target> parameter

    ...

    If <target> is EGL_LINUX_DMA_BUF_EXT, <dpy> must be a valid display, <ctx>
    must be EGL_NO_CONTEXT, and <buffer> must be NULL, cast into the type
    EGLClientBuffer. The details of the image is specified by the attributes
    passed into eglCreateImageKHR. Required attributes and their values are as
    follows:

        * EGL_WIDTH & EGL_HEIGHT: The logical dimensions of the buffer in pixels

        * EGL_LINUX_DRM_FOURCC_EXT: The pixel format of the buffer, as specified
          by drm_fourcc.h and used as the pixel_format parameter of the
          drm_mode_fb_cmd2 ioctl.

        * EGL_DMA_BUF_PLANE0_FD_EXT: The dma_buf file descriptor of plane 0 of
          the image.

        * EGL_DMA_BUF_PLANE0_OFFSET_EXT: The offset from the start of the
          dma_buf of the first sample in plane 0, in bytes.

        * EGL_DMA_BUF_PLANE0_PITCH_EXT: The number of bytes between the start of
          subsequent rows of samples in plane 0. May have special meaning for
          non-linear formats.

    For images in an RGB color-space or those using a single-plane YUV format,
    only the first plane's file descriptor, offset & pitch should be specified.
    For semi-planar YUV formats, that first plane (plane 0) holds only the luma
    samples and chroma samples are stored interleaved in a second plane (plane
    1). For fully planar YUV formats, the first plane (plane 0) continues to
    hold the luma samples however the chroma samples are stored seperately in
    two additional planes (plane 1 & plane 2). If present, planes 1 & 2 are
    specified by the following attributes, which have the same meanings as
    defined above for plane 0:

        * EGL_DMA_BUF_PLANE1_FD_EXT
        * EGL_DMA_BUF_PLANE1_OFFSET_EXT
        * EGL_DMA_BUF_PLANE1_PITCH_EXT
        * EGL_DMA_BUF_PLANE2_FD_EXT
        * EGL_DMA_BUF_PLANE2_OFFSET_EXT
        * EGL_DMA_BUF_PLANE2_PITCH_EXT

    The ordering of samples within a plane is taken from the drm_fourcc
    pixel_format specified for EGL_LINUX_DRM_FOURCC_EXT. For example, if
    EGL_LINUX_DRM_FOURCC_EXT is set to DRM_FORMAT_NV12, the chroma plane
    specified by EGL_DMA_BUF_PLANE1* contains samples in the order V, U,
    whereas if EGL_LINUX_DRM_FOURCC_EXT is DRM_FORMAT_NV21, the order is U,
    V. Similarly, the ordering of planes for fully-planar formats is also taken
    from the pixel_format specified as EGL_LINUX_DRM_FOURCC_EXT. For example,
    if EGL_LINUX_DRM_FOURCC_EXT is set to DRM_FORMAT_YUV410, the luma plane is
    specified by EGL_DMA_BUF_PLANE0*, the plane containing U-samples is
    specified by EGL_DMA_BUF_PLANE1* and the plane containing the V-samples is
    specified by EGL_DMA_BUF_PLANE2*, whereas if EGL_LINUX_DRM_FOURCC_EXT is
    set to DRM_FORMAT_YVU410, plane 1 contains the V-samples and plane 2
    contains the U-samples.

    In addition to the above required attributes, the application may also
    provide hints as to how the data should be interpreted by the GL. If any of
    these hints are not specified, the GL will guess based on the pixel format
    passed as the EGL_LINUX_DRM_FOURCC_EXT attribute or may fall-back to some
    default value. Not all GLs will be able to support all combinations of
    these hints and are free to use whatever settings they choose to achieve
    the closest possible match.

        * EGL_YUV_COLOR_SPACE_HINT_EXT: The color-space the data is in. Only
          relevant for images in a YUV format, ignored when specified for an
          image in an RGB format. Accepted values are:
          EGL_ITU_REC601_EXT, EGL_ITU_REC709_EXT & EGL_ITU_REC2020_EXT.

        * EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT &
          EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT: Where chroma samples are
          sited relative to luma samples when the image is in a sub-sampled
          format. When the image is not using chroma sub-sampling, the luma and
          chroma samples are assumed to be co-sited. Siting is split into the
          vertical and horizontal and is in a fixed range. A siting of zero
          means the first luma sample is taken from the same position in that
          dimension as the chroma sample. This is best illustrated in the
          diagram below:

                 (0.5, 0.5)        (0.0, 0.5)        (0.0, 0.0)
                +   +   +   +     +   +   +   +     *   +   *   +
                  x       x       x       x
                +   +   +   +     +   +   +   +     +   +   +   +

                +   +   +   +     +   +   +   +     *   +   *   +
                  x       x       x       x
                +   +   +   +     +   +   +   +     +   +   +   +

            Luma samples (+), Chroma samples (x) Chrome & Luma samples (*)

          Note this attribute is ignored for RGB images and non sub-sampled
          YUV images. Accepted values are: EGL_YUV_CHROMA_SITING_0_EXT (0.0)
          & EGL_YUV_CHROMA_SITING_0_5_EXT (0.5)

        * EGL_SAMPLE_RANGE_HINT_EXT: The numerical range of samples. Only
          relevant for images in a YUV format, ignored when specified for
          images in an RGB format. Accepted values are: EGL_YUV_FULL_RANGE_EXT
          (0-256) & EGL_YUV_NARROW_RANGE_EXT (16-235).


    If eglCreateImageKHR is successful for a EGL_LINUX_DMA_BUF_EXT target, the
    EGL will take a reference to the dma_buf(s) which it will release at any
    time while the EGLDisplay is initialized. It is the responsibility of the
    application to close the dma_buf file descriptors."


    Add to the list of error conditions for eglCreateImageKHR:

      "* If <target> is EGL_LINUX_DMA_BUF_EXT and <buffer> is not NULL, the
         error EGL_BAD_PARAMETER is generated.

       * If <target> is EGL_LINUX_DMA_BUF_EXT, and the list of attributes is
         incomplete, EGL_BAD_PARAMETER is generated.

       * If <target> is EGL_LINUX_DMA_BUF_EXT, and the EGL_LINUX_DRM_FOURCC_EXT
         attribute is set to a format not supported by the EGL, EGL_BAD_MATCH
         is generated.

       * If <target> is EGL_LINUX_DMA_BUF_EXT, and the EGL_LINUX_DRM_FOURCC_EXT
         attribute indicates a single-plane format, EGL_BAD_ATTRIBUTE is
         generated if any of the EGL_DMA_BUF_PLANE1_* or EGL_DMA_BUF_PLANE2_*
         attributes are specified.

       * If <target> is EGL_LINUX_DMA_BUF_EXT and the value specified for
         EGL_YUV_COLOR_SPACE_HINT_EXT is not EGL_ITU_REC601_EXT,
         EGL_ITU_REC709_EXT or EGL_ITU_REC2020_EXT, EGL_BAD_ATTRIBUTE is
         generated.

       * If <target> is EGL_LINUX_DMA_BUF_EXT and the value specified for
         EGL_SAMPLE_RANGE_HINT_EXT is not EGL_YUV_FULL_RANGE_EXT or
         EGL_YUV_NARROW_RANGE_EXT, EGL_BAD_ATTRIBUTE is generated.

       * If <target> is EGL_LINUX_DMA_BUF_EXT and the value specified for
         EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT or
         EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT is not
         EGL_YUV_CHROMA_SITING_0_EXT or EGL_YUV_CHROMA_SITING_0_5_EXT,
         EGL_BAD_ATTRIBUTE is generated.

       * If <target> is EGL_LINUX_DMA_BUF_EXT and one or more of the values
         specified for a plane's pitch or offset isn't supported by EGL,
         EGL_BAD_ACCESS is generated.


Issues

    1. Should this be a KHR or EXT extension?

    ANSWER: EXT. Khronos EGL working group not keen on this extension as it is
    seen as contradicting the EGLStream direction the specification is going in.
    The working group recommends creating additional specs to allow an EGLStream
    producer/consumer connected to v4l2/DRM or any other Linux interface.

    2. Should this be a generic any platform extension, or a Linux-only
    extension which explicitly states the handles are dma_buf fds?

    ANSWER: There's currently no intention to port this extension to any OS not
    based on the Linux kernel. Consequently, this spec can be explicitly written
    against Linux and the dma_buf API.

    3. Does ownership of the file descriptor pass to the EGL library?

    ANSWER: No, EGL does not take ownership of the file descriptors. It is the
    responsibility of the application to close the file descriptors on success
    and failure.

    4. How are the different YUV color spaces handled (BT.709/BT.601)?

    ANSWER: The pixel formats defined in drm_fourcc.h only specify how the data
    is laid out in memory. It does not define how that data should be
    interpreted. Added a new EGL_YUV_COLOR_SPACE_HINT_EXT attribute to allow the
    application to specify which color space the data is in to allow the GL to
    choose an appropriate set of co-efficients if it needs to convert that data
    to RGB for example.

    5. What chroma-siting is used for sub-sampled YUV formats?

    ANSWER: The chroma siting is not specified by either the v4l2 or DRM APIs.
    This is similar to the color-space issue (4) in that the chroma siting
    doesn't affect how the data is stored in memory. However, the GL will need
    to know the siting in order to filter the image correctly. While the visual
    impact of getting the siting wrong is minor, provision should be made to
    allow an application to specify the siting if desired. Added additional
    EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT &
    EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT attributes to allow the siting to
    be specified using a set of pre-defined values (0 or 0.5).

    6. How can an application query which formats the EGL implementation
    supports?

    PROPOSAL: Don't provide a query mechanism but instead add an error condition
    that EGL_BAD_MATCH is raised if the EGL implementation doesn't support that
    particular format.

    7. Which image formats should be supported and how is format specified?

    Seem to be two options 1) specify a new enum in this specification and
    enumerate all possible formats. 2) Use an existing enum already in Linux,
    either v4l2_mbus_pixelcode and/or those formats listed in drm_fourcc.h?

    ANSWER: Go for option 2) and just use values defined in drm_fourcc.h.

    8. How can AYUV images be handled?

    ANSWER: At least on fourcc.org and in drm_fourcc.h, there only seems to be
    a single AYUV format and that is a packed format, so everything, including
    the alpha component would be in the first plane.

    9. How can you import interlaced images?

    ANSWER: Interlaced frames are usually stored with the top & bottom fields
    interleaved in a single buffer. As the fields would need to be displayed as
    at different times, the application would create two EGLImages from the same
    buffer, one for the top field and another for the bottom. Both EGLImages
    would set the pitch to 2x the buffer width and the second EGLImage would use
    a suitable offset to indicate it started on the second line of the buffer.
    This should work regardless of whether the data is packed in a single plane,
    semi-planar or multi-planar.

    If each interlaced field is stored in a separate buffer then it should be
    trivial to create two EGLImages, one for each field's buffer.

    10. How are semi-planar/planar formats handled that have a different
    width/height for Y' and CbCr such as YUV420?

    ANSWER: The spec says EGL_WIDTH & EGL_HEIGHT specify the *logical* width and
    height of the buffer in pixels. For pixel formats with sub-sampled Chroma
    values, it should be trivial for the EGL implementation to calculate the
    width/height of the Chroma sample buffers using the logical width & height
    and by inspecting the pixel format passed as the EGL_LINUX_DRM_FOURCC_EXT
    attribute. I.e. If the pixel format says it's YUV420, the Chroma buffer's
    width = EGL_WIDTH/2 & height =EGL_HEIGHT/2.

    11. How are Bayer formats handled?

    ANSWER: As of Linux 2.6.34, drm_fourcc.h does not include any Bayer formats.
    However, future kernel versions may add such formats in which case they
    would be handled in the same way as any other format.

    12. Should the spec support buffers which have samples in a "narrow range"?

    Content sampled from older analogue sources typically don't use the full
    (0-256) range of the data type storing the sample and instead use a narrow
    (16-235) range to allow some headroom & toeroom in the signals to avoid
    clipping signals which overshoot slightly during processing. This is
    sometimes known as signals using "studio swing".

    ANSWER: Add a new attribute to define if the samples use a narrow 16-235
    range or the full 0-256 range.

    13. Specifying the color space and range seems cumbersome, why not just
    allow the application to specify the full YUV->RGB color conversion matrix?

    ANSWER: Some hardware may not be able to use an arbitrary conversion matrix
    and needs to select an appropriate pre-defined matrix based on the color
    space and the sample range.

    14. How do you handle EGL implementations which have restrictions on pitch
    and/or offset?

    ANSWER: Buffers being imported using dma_buf pretty much have to be
    allocated by a kernel-space driver. As such, it is expected that a system
    integrator would make sure all devices which allocate buffers suitable for
    exporting make sure they use a pitch supported by all possible importers.
    However, it is still possible eglCreateImageKHR can fail due to an
    unsupported pitch. Added a new error to the list indicating this.

    15. Should this specification also describe how to export an existing
    EGLImage as a dma_buf file descriptor?

    ANSWER: No. Importing and exporting buffers are two separate operations and
    importing an existing dma_buf fd into an EGLImage is useful functionality in
    itself. Agree that exporting an EGLImage as a dma_buf fd is useful, E.g. it
    could be used by an OpenMAX IL implementation's OMX_UseEGLImage function to
    give access to the buffer backing an EGLImage to video hardware. However,
    exporting can be split into a separate extension specification.


Revision History

#7 (Kristian H. Kristensen, December 13, 2017)
   - Clarify plane ordering to match Linux FOURCC conventions (Bug 16017).

#6 (David Garbett, December 05, 2013)
   - Application now retains ownership of dma_buf file descriptors.

#5 (Tom Cooksey, February 19, 2013)
   - Assigned enum values
   - Moved out of drafts

#4 (Tom Cooksey, October 04, 2012)
   - Fixed issue numbering!
   - Added issues 8 - 15.
   - Promoted proposal for Issue 3 to be the answer.
   - Added an additional attribute to allow an application to specify the color
     space as a hint which should address issue 4.
   - Added an additional attribute to allow an application to specify the chroma
     siting as a hint which should address issue 5.
   - Added an additional attribute to allow an application to specify the sample
     range as a hint which should address the new issue 12.
   - Added language to end of error section clarifying who owns the fd passed
     to eglCreateImageKHR if an error is generated.

#3 (Tom Cooksey, August 16, 2012)
   - Changed name from EGL_EXT_image_external and re-written language to
     explicitly state this for use with Linux & dma_buf.
   - Added a list of issues, including some still open ones.

#2 (Jesse Barker, May 30, 2012)
   - Revision to split eglCreateImageKHR functionality from export
     Functionality.
   - Update definition of EGLNativeBufferType to be a struct containing a list
     of handles to support multi-buffer/multi-planar formats.

#1 (Jesse Barker, March 20, 2012)
   - Initial draft.
