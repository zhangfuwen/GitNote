# KHR_image_base

Name

    KHR_image_base

Name Strings

    EGL_KHR_image_base

Contributors

    Jeff Juliano
    Gary King
    Jon Leech
    Jonathan Grant
    Barthold Lichtenbelt
    Aaftab Munshi
    Acorn Pooley
    Chris Wynn

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)
    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright (c) 2008-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete. Functionality approved (as part of KHR_image) by the
    Khronos Board of Promoters on February 11, 2008.

    Split into KHR_image_base and KHR_image_pixmap approved by the
    Khronos Technical Working Group on November 19, 2008. Update to
    version 5 approved on December 10, 2008.

Version

    Version 8, August 27, 2014

Number

    EGL Extension #8

Dependencies

    EGL 1.2 is required.

    An EGL client API, such as OpenGL ES or OpenVG, is required.

    This extension is written against the wording of the EGL 1.2
    Specification.

Overview

    This extension defines a new EGL resource type that is suitable for
    sharing 2D arrays of image data between client APIs, the EGLImage.
    Although the intended purpose is sharing 2D image data, the
    underlying interface makes no assumptions about the format or
    purpose of the resource being shared, leaving those decisions to
    the application and associated client APIs.

Glossary

    EGLImage:  An opaque handle to a shared resource created by EGL
               client APIs, presumably a 2D array of image data

    EGLImage source:  An object or sub-object originally created in
               a client API (such as a mipmap level of a texture object
               in OpenGL-ES, or a VGImage in OpenVG) which is used as
               the <buffer> parameter in a call to eglCreateImageKHR.

    EGLImage target:  An object created in a client API (such as a
               texture object in OpenGL-ES or a VGImage in OpenVG)
               from a previously-created EGLImage

    EGLImage sibling: The set of all EGLImage targets (in all
               client API contexts) which are created from the
               same EGLImage object, and the EGLImage source resouce
               which was used to create that EGLImage.

    Orphaning:  The process of respecifying and/or deleting an EGLImage
               sibling resource (inside a client API context) which
               does not result in deallocation of the memory associated
               with the EGLImage or affect rendering results using other
               EGLImage siblings.

    Referencing:  The process of creating an EGLImage target resource
               (inside a client API context) from an EGLImage.

    Respecification: When the size, format, or other attributes of an
               EGLImage sibling are changed via client API calls such as
               gl*TexImage*. Respecification usually will result in
               orphaning the sibling. Note that changing the pixel values of
               the sibling (e.g. by rendering to it or by calling
               gl*TexSubImage*) does not constitute respecification.

New Types

    /*
     * EGLImageKHR is an object which can be used to create EGLImage
     * target resources (inside client APIs).
     */
    typedef void* EGLImageKHR;

New Procedures and Functions

```c
    EGLImageKHR eglCreateImageKHR(
                            EGLDisplay dpy,
                            EGLContext ctx,
                            EGLenum target,
                            EGLClientBuffer buffer,
                            const EGLint *attrib_list)

    EGLBoolean eglDestroyImageKHR(
                            EGLDisplay dpy,
                            EGLImageKHR image)
```                            

New Tokens

    Returned by eglCreateImageKHR:

        EGL_NO_IMAGE_KHR                                     ((EGLImageKHR)0)

    Accepted as an attribute in the <attrib_list> parameter of
    eglCreateImageKHR:

        EGL_IMAGE_PRESERVED_KHR                              0x30D2

Additions to Chapter 2 of the EGL 1.2 Specification (EGL Operation)

    Add a new section "EGLImages" after section 2.4:

    "2.5 EGLImages

    As described in section 2.4, EGL allows contexts of the same client
    API type to share significant amounts of state (such as OpenGL-ES
    texture objects and OpenVG paths); however, in some cases it may
    be desirable to share state between client APIs - an example would be
    using a previously-rendered OpenVG image as an OpenGL-ES texture
    object.

    In order to facilitate these more complicated use-cases, EGL is capable
    of creating EGL resources that can be shared between contexts of
    different client APIs (called "EGLImages") from client API resources
    such as texel arrays in OpenGL-ES texture objects or OpenVG VGImages
    (collectively, the resources that are used to create EGLImages are
    referred to as "EGLImage sources").

    The EGL client APIs each provide mechanisms for creating appropriate
    resource types (such as complete texture arrays or OpenVG VGImages) from
    EGLImages through a API-specific mechanisms.  Collectively, resources
    which are created from EGLImages within client APIs are referred to as
    "EGLImage targets."  Each EGLImage may have multiple associated EGLImage
    targets.  Collectively, the EGLImage source and EGLImage targets
    associated with an EGLImage object are referred to as "EGLImage
    siblings."

    2.5.1  EGLImage Specification

    The command

        EGLImageKHR eglCreateImageKHR(
                                EGLDisplay dpy,
                                EGLContext ctx,
                                EGLenum target,
                                EGLClientBuffer buffer,
                                const EGLint *attrib_list)

    is used to create an EGLImage from an existing image resource <buffer>.
    <dpy> specifies the EGL display used for this operation.
    <ctx> specifies the EGL client API context
    used for this operation, or EGL_NO_CONTEXT if a client API context is not
    required.  <target> specifies the type of resource being used as the
    EGLImage source (examples include two-dimensional textures in OpenGL ES
    contexts and VGImage objects in OpenVG contexts).  <buffer> is the name
    (or handle) of a resource to be used as the EGLImage source, cast into the
    type EGLClientBuffer.  <attrib_list> is an list of attribute-value pairs
    which is used to select sub-sections of <buffer> for use as the EGLImage
    source, such as mipmap levels for OpenGL ES texture map resources, as well as
    behavioral options, such as whether to preserve pixel data during creation. If
    <attrib_list> is non-NULL, the last attribute specified in the list must
    be EGL_NONE.

    The resource specified by <dpy>, <ctx>, <target>, <buffer>, and
    <attrib_list> must not itself be an EGLImage sibling, or bound to an EGL
    PBuffer resource (eglBindTexImage, eglCreatePbufferFromClientBuffer).

    Values accepted for <target> are listed in Table aaa, below(fn1).
        (fn1) No values are defined by this extension. All functionality
        to create EGLImages from other types of resources, such as
        native pixmaps, GL textures, and VGImages, is layered in other
        extensions.

      +-------------------------+--------------------------------------------+
      |  <target>               |  Notes                                     |
      +-------------------------+--------------------------------------------+
      +-------------------------+--------------------------------------------+
       Table aaa.  Legal values for eglCreateImageKHR <target> parameter

    Attribute names accepted in <attrib_list> are shown in Table bbb,
    together with the <target> for which each attribute name is valid, and
    the default value used for each attribute if it is not included in
    <attrib_list>.

      +-------------------------+----------------------+-----------+---------------+
      | Attribute               | Description          | Valid     | Default Value |
      |                         |                      | <target>s |               |
      +-------------------------+----------------------+-----------+---------------+
      | EGL_NONE                | Marks the end of the | All       | N/A           |
      |                         | attribute-value list |           |               |
      | EGL_IMAGE_PRESERVED_KHR | Whether to preserve  | All       | EGL_FALSE     |
      |                         | pixel data           |           |               |
      +-------------------------+----------------------+-----------+---------------+
       Table bbb.  Legal attributes for eglCreateImageKHR <attrib_list> parameter

    This command returns an EGLImageKHR object corresponding to the image
    data specified by <dpy>, <ctx>, <target>, <buffer> and <attrib_list> which
    may be referenced by client API operations, or EGL_NO_IMAGE_KHR in the
    event of an error.

    If the value of attribute EGL_IMAGE_PRESERVED_KHR is EGL_FALSE (the
    default), then all pixel data values associated with <buffer> will be
    undefined after eglCreateImageKHR returns.

    If the value of attribute EGL_IMAGE_PRESERVED_KHR is EGL_TRUE, then all
    pixel data values associated with <buffer> are preserved.

    Errors

        If eglCreateImageKHR fails, EGL_NO_IMAGE_KHR will be returned, the
        contents of <buffer> will be unaffected, and one of the following
        errors will be generated:

       * If <dpy> is not the handle of a valid EGLDisplay object, the error
         EGL_BAD_DISPLAY is generated.

       * If <ctx> is neither the handle of a valid EGLContext object on
         <dpy> nor EGL_NO_CONTEXT, the error EGL_BAD_CONTEXT is
         generated.

       * If <target> is not one of the values in Table aaa, the error
         EGL_BAD_PARAMETER is generated.

       * If an attribute specified in <attrib_list> is not one of the
         attributes listed in Table bbb, the error EGL_BAD_PARAMETER is
         generated.

       * If an attribute specified in <attrib_list> is not a valid attribute
         for <target>, as shown in Table bbb, the error EGL_BAD_MATCH is
         generated.

       * If the resource specified by <dpy>, <ctx>, <target>, <buffer> and
         <attrib_list> has an off-screen buffer bound to it (e.g., by a
         previous call to eglBindTexImage), the error EGL_BAD_ACCESS is
         generated.

       * If the resource specified by <dpy>, <ctx>, <target>, <buffer> and
         <attrib_list> is bound to an off-screen buffer (e.g., by a previous
         call to eglCreatePbufferFromClientBuffer), the error
         EGL_BAD_ACCESS is generated.

       * If the resource specified by <dpy>, <ctx>, <target>, <buffer> and
         <attrib_list> is itself an EGLImage sibling, the error
         EGL_BAD_ACCESS is generated.

       * If insufficient memory is available to complete the specified
         operation, the error EGL_BAD_ALLOC is generated.

       * If the call to eglCreateImageKHR fails for multiple reasons, the
         generated error must be appropriate for one of the reasons,
         although the specific error returned is undefined.

       * If the value specified in <attrib_list> for EGL_IMAGE_PRESERVED_KHR
         is EGL_TRUE, and an EGLImageKHR handle cannot be created from the
         specified resource such that the pixel data values in <buffer> are
         preserved, the error EGL_BAD_ACCESS is generated.

    Note that the success or failure of eglCreateImageKHR should not affect
    the ability to use <buffer> in its original API context (or context
    share group) (although the pixel data values will be undefined if
    EGL_IMAGE_PRESERVED_KHR is not EGL_TRUE).

    2.5.2  Lifetime and Usage of EGLImages

    Once an EGLImage is created from an EGLImage source, the memory associated
    with the EGLImage source will remain allocated (and all EGLImage siblings
    in all client API contexts will be useable) as long as either of the
    following conditions is true:
      A)  Any EGLImage siblings exist in any client API context
      B)  The EGLImage object exists inside EGL

    The semantics for specifying, deleting and using EGLImage siblings are
    client API-specific, and are described in the appropriate API
    specifications.

    If an application specifies an EGLImage sibling as the destination for
    rendering and/or pixel download operations (e.g., as an OpenGL-ES
    framebuffer object, glTexSubImage2D, etc.), the modified image results
    will be observed by all EGLImage siblings in all client API contexts.
    If multiple client API contexts access EGLImage sibling resources
    simultaneously, with one or more context modifying the image data,
    rendering results in all contexts accessing EGLImage siblings are
    undefined.

    Respecification and/or deletion of any EGLImage sibling (i.e., both
    EGLImage source and EGLImage target resources) inside a client API
    context (e.g., by issuing a subsequent call to
    gl{Copy,Compressed}TexImage, glDeleteTextures, with the EGLImage
    sibling resource as the target of the operation) affects only that
    client API context and other contexts within its share group.  The
    specific semantics for this behavior are defined by each client API,
    and generally results in orphaning of the EGLImage, and may also
    include allocation of additional memory for the respecified resource
    and/or copying of the EGLImage pixel data.

    Operations inside EGL or any client API context which may affect the
    lifetime of an EGLImage (or the memory allocated for the EGLImage),
    such as respecifying and/or deleting an EGLImage sibling inside a
    client API context, must be atomic.

    Applications may create client API resources from an EGLImageKHR using
    client API extensions outside the scope of this document (such as
    GL_OES_EGL_image, which creates OpenGL ES texture and renderbuffer
    objects). If the EGLImageKHR used to create the client resource was
    created with the EGL_IMAGE_PRESERVED_KHR attribute set to EGL_TRUE, then
    the pixel data values associated with the image will be preserved after
    creating the client resource; otherwise, the pixel data values will be
    undefined. If the EGLImageKHR was created with the
    EGL_IMAGE_PRESERVED_KHR attribute set to EGL_TRUE, and EGL is unable to
    create the client resource without modifying the pixel values, then
    creation will fail and the pixel data values will be preserved.

    The command

        EGLBoolean eglDestroyImageKHR(
                            EGLDisplay dpy,
                            EGLImageKHR image)

    is used to destroy the specified EGLImageKHR object <image>.  Once
    destroyed, <image> may not be used to create any additional EGLImage
    target resources within any client API contexts, although existing
    EGLImage siblings may continue to be used.  EGL_TRUE is returned
    if DestroyImageKHR succeeds, EGL_FALSE indicates failure.

       * If <dpy> is not the handle of a valid EGLDisplay object, the error
         EGL_BAD_DISPLAY is generated.

       * If <image> is not a valid EGLImageKHR object created with respect
         to <dpy>, the error EGL_BAD_PARAMETER is generated."

    Add a new error to the list at the bottom of Section 3.5.3 (Binding
    Off-Screen Rendering Surfaces to Client Buffers):

       "* If the buffers contained in <buffer> consist of any EGLImage
          siblings, an EGL_BAD_ACCESS error is generated."

Issues

    1.  What resource types should be supported by this extension?

        RESOLVED:  This specification is designed to support the
        sharing of two-dimensional image resources between client APIs,
        as these resources are a fundamental component of all modern
        graphics APIs.

        Other resources types (e.g., buffer objects) will not be directly
        supported by this specification, due to a variety of reasons:

            a.  An absense of use cases for this functionality
            b.  Handling the semantics for some of these resources
                (e.g., glMapBuffer) would significantly complicate
                and delay this specification.
            c.  A desire to address the image-sharing use cases
                as quickly as possible.

        Should additional resource-sharing functionality be desired
        in the future, the framework provided by this specification
        should be extendable to handle more general resource
        sharing.

    2.  Should this specification address client API-specific resources
        (OpenGL texture maps, OpenVG VGImages), or should that
        functionality be provided by layered extensions?

        SUGGESTION: Use layered extensions, even for for sharing image
        data with native rendering APIs (the EGL_KHR_image_pixmap
        extension).

        There are two major arguments for using layered extensions:

          1.  The two client APIs which are defined at the time of this
              specification (OpenVG, OpenGL ES) may not always be
              deployed on a device; many devices may choose to implement
              just one of these two APIs.  However, even single-API
              devices may benefit from the ability to share image data
              with native rendering APIs (provided in this specification)
              or with the OpenMAX API.

          2.  OpenGL ES defines a number of optional resource types
              (cubemaps, renderbuffers, volumetric textures) which this
              framework should support; however, implementations may not.
              By layering each of these resource types in individual
              extensions, implementations which are limited to just the
              core OpenGL ES 1.1 (or OpenGL ES 2.0) features will not
              need to add EGLImage enumerant support for unsupported
              resource types.

        The original EGL_KHR_image extension included native pixmap
        functionality. We have now split the abstract base functionality
        (the egl{Create,Destroy}ImageKHR APIs) from the native pixmap
        functionality, and redefined EGL_KHR_image as the combination of
        EGL_KHR_image_base and EGL_KHR_image_pixmap.

    3.  Should attributes (width, height, format, etc.) for EGLImages
        be queriable?

        SUGGESTION:  No.  Given the wealth of attributes that we would
        need to specify all possible EGLImages (and possible
        memory layout optimizations performed by implementations), we
        can dramatically simplify the API without loss of key
        functionality by making EGLImages opaque and allowing
        implementations to make the correct decisions internally.

    4.  Should this specification allow the creation of EGLImages from
        client API resources which are themselves EGLImage targets?

        RESOLVED:  No.  This can make memory garbage collection and
        reference counting more difficult, with no practical benefit.
        Instead, generate an error if an application attempts to
        create an EGLImage from an EGLImage target resource.

    5.  Should this specification allow multiple EGLImages to be created
        from the same EGLImage source resource?

        RESOLVED:  No.  The resource <buffer> specified to
        eglCreateImageKHR may include multiple sub-objects; examples are
        mipmapped images and cubemaps in the OpenGL-ES API.  However, the
        EGLImage source is defined as the specific sub-object that is defined
        by: <ctx>, <target>, <buffer>, and <attrib_list>.  This sub-object must
        not be an EGLImage sibling (either EGLImage source or EGLImage target)
        when eglCreateImageKHR is called; however, other sub-objects in
        <buffer> may be EGLImage siblings.  This allows applications to share
        individual cubemap faces, or individual mipmap levels of detail across
        all of the supported APIs.

        Note that the EGLImage source and any EGLImage target resources
        will still be EGLImage siblings, even if the EGLImage object
        is destroyed by a call to DestroyImageKHR.

    6.  If an EGLImage sibling is respecified (or deleted), what
        should happen to the EGLImage and any other EGLImage
        siblings?

        RESOLVED:  The principle of least surprise would dictate that
        respecification and/or deletion of a resource in one client API
        should not adversely affect operation in other client APIs
        (such as introducing errors).

        Applying this to EGLImages, respecification and/or deletion
        of one EGLImage sibling should not respecify/delete other
        EGLImage siblings.  Each client API will be responsible for
        defining appropriate semantics to meet this restriction;
        however, example behaviors may include one or more of:
        allocating additional memory for the respecified resource,
        deleting the EGLImage sibling resource without deallocating
        the associated memory ("orphaning") and/or copying the
        existing EGLImage pixel data to an alternate memory location.

        The memory associated with EGLImage objects should remain
        allocated as long as any EGLImage sibling resources exist
        in any client API context.

    7.  Should this specification address synchronization issues
        when multiple client API contexts simultaneously access EGLImage
        sibling resources?

        RESOLVED:  No.  Including error-producing lock and synchronization
        semantics would introduce additional (undesirable) validation
        overhead in numerous common operations (e.g., glBindTexture,
        glDrawArrays, etc.).  Rather than burdening implementations (and
        applications) with this overhead, a separate synchronization
        mechanism should be exposed to applications.

    8.  Should eglCreatePbufferFromClientBuffer accept buffer parameters
        which are EGLImage siblings?

        RESOLVED:  No.  Allowing this behavior creates very complex
        circular dependency possibilities (CreateImage / DeriveImage /
        CreatePbufferFromClientBuffer / BindTexImage /
        CreateImage / ...) with no practical benefit.  Therefore,
        attempting to create a Pbuffer from a client buffer which
        is an EGLImage sibling should generate an error.

    9.  Should CreateImage accept client buffers which are bound to
        Pbuffers (through eglBindTexImage)?

        RESOLVED:  No, for the same reasons listed in Issue 8.

    10. Should implementations be allowed to modify the pixel data in the
        EGLImage source buffers specified to eglCreateImageKHR?

        SUGGESTION:  By allowing previously-existing image data to become
        undefined after calls to eglCreateImageKHR, implementations are able
        to perform any necessary reallocations required for cross-API
        buffer compatibility (and/or performance), without requiring
        copy-aside functionality.  Because applications are able to
        respecify the pixel data through mechanisms such as vgSubImage
        and glTexSubImage, no use-cases are restricted by this.

        Therefore, the current suggestion is to allow implementations
        to leave pixel data undefined after calls to eglCreateImageKHR
        functions.  The current spec revision has been written in
        this way.

    11. What is the correct mechanism for specifying the EGLImage source
        resources used to create an EGLImage object?

        RESOLVED:  Three different mechanisms were discussed while
        defining this extension:

            A)  Providing resource-specific creation functions, such as
                eglCreateImage2DKHR, eglCreateImage3DKHR, etc.

            B)  Providing a single creation function which returns a
                "NULL" EGLImage object, and requiring client APIs to
                define additional functions which would allow client API
                resources to be "bound" to the EGLImage object.

            C)  Provide a single resource creation function, and use
                an attribute-value list with attributes specific to the
                "target" image resource.

        Initial specifications were written using Option (A); however,
        it was believed that this structure would result in an increase
        in the number of entry points over time as additional client APIs
        and client API resource targets were added.  Furthermore, reuse
        of these functions was resulting in cases where parameters were
        required to have modal behavior: a 2D image creation function
        was required to have a mipmap level of detail parameter for
        OpenGL ES texture maps, but this same parameter would need to be
        0 for OpenVG.

        Option (B) provided some nice characteristics: as client APIs
        continue to evolve, any extensions needed to allow EGLImage
        creation could be isolated in the individual client API, rather
        than necessitating an EGL extension.  However, the creation of
        "NULL" images created additional semantic complexity and error
        conditions (e.g., attempting to derive an EGLImage target from a
        "NULL" image), and every client API would need to provide a
        function for every unique resource type; instead of one common
        API function for pixmap, OpenGL 2D textures, and OpenVG VGImages,
        three would be required.

        This specification is written using Option (C).  There is a
        single CreateImage function, with a <target> parameter defining
        the EGLImage source type, and an attribute-value list allowing
        for additional selection of resource sub-sections.  This
        maximizes entry-point reuse, and minimizes the number of
        redundant parameters an application may be required to send.
        This framework allows for layered extensions to be easily
        written, so little churn is expected as client APIs evolve.

    12. Should a context be explicitly provided to eglCreateImageKHR,
        or should the context be deduced from the current thread's
        bound API?

        SUGGESTION:  For clarity (both in usage and spec language), the
        context containing the EGLImage source should be provided by the
        application, rather than inferring the context from EGL state.

    13. Why does this extension define a new EGL object type, rather
        than using the existing EGLSurface objects?

        RESOLVED:  Although controversial, the creation of a new,
        opaque image object type removes several fundamental problems
        with the EGLSurface (and Pbuffer) API:

            1)  The tight compatibility requirements of EGLSurfaces
                and EGLConfigs necessitated applications creating
                (and calling MakeCurrent) for every unique pixel
                format used during rendering.  This has already caused
                noticeable performance problems in OpenGL-ES (and
                desktop OpenGL), and is the primary reason that
                framebuffer objects were created.

            2)  Application use-cases are centered around sharing of
                color image data, although unique "sundry" buffers
                (such as depth, stencil and alpha mask) may be used
                in each client API.

            3)  Extending the CreatePbuffer interface to support fully-
                specifying all possible buffer attributes in all client
                APIs will become unwieldy, particularly as new EGL
                client APIs and pixel formats are introduced.

        The EGLImage proposal addresses all three of these restrictions:

        1) is addressed by placing the burden of framebuffer management
        inside the client API, and allowing EGLImages to be accessed
        inside client APIs using an appropriate resource type (such
        as OpenGL-ES renderbuffers).  This follows the example provided
        by the GL_OES_framebuffer_object specification.

        2) is addressed by defining EGLImages to be "trivial" two-
        dimensional arrays of pixel data.  Implementations may choose
        to support creation of EGLImages from any type of pixel data,
        and the association of multiple EGLImages and/or sundry
        buffers into a single framebuffer is the responsibility of the
        application and client API, using a mechanism such as
        GL_OES_framebuffer_object.

        3) is addressed by defining EGLImages as opaque and
        non-queriable.  Although this introduces potential portability
        problems (addressed separately in issue 15), it avoids the
        ever-expanding problem of defining buffer compatibility as the
        cartesian product of all possible buffer attributes.

    14. Since referencing EGLImages is the responsibility of the client
        API, and may fail for implementation-dependent reasons,
        doesn't this result in a potential portability problem?

        UNRESOLVED:  Yes, this portability problem (where referencing
        succeeds on one platform but generates errors on a different
        one) is very similar to the implementation-dependent
        failure introduced in the EXT_framebuffer_object specification,
        discussed (at length) in Issues (12), (37), (46), (48) and (61)
        of that specification.  Similar to that specification, this
        specification should include some "minimum requirements"
        language for EGLImage creation and referencing.

        Since there are numerous references to an upcoming
        "format restriction" API in the EXT_framebuffer_object
        specification, it may be valuable to wait until that API is
        defined before attempting to define a similar API for
        EGLImages.

    15. Should creation of an EGLImage from an EGLImage source
        introduce the possibility for errors in the EGLImage source's
        owning context?

        RESOLVED:  No; although image data may be undefined (issue 11),
        the (successful or unsuccessful) creation of an EGLImage should
        not introduce additional error conditions in the EGLImage
        source's owning context.  Text added to the end of section
        2.5.1 describing this.

    16. Is it reasonable to require that when a preserved EGLImage is
        used by layered extensions to create client API siblings of that
        image, pixel data values are preserved?

        UNRESOLVED: There are at least two extensions that reference
        EGLImages to create EGLImage targets, VG_KHR_EGL_image and
        GL_OES_EGL_image.

        Each of these extensions makes provision for failing the creation of
        the EGLImage target due to "an implementation-dependent reason".
        This could include that the pixel data has been marked as preserved,
        and that the implementation is not able to create the EGLImage
        target without causing the pixel data of the original EGLImage
        source <buffer> to become undefined.

        Issue 14 of EGL_KHR_image also discusses the consequences of failure
        for implementation-dependent reasons. This implies that all
        extensions for referencing an EGLImage need to make provision for
        implementation-dependent failure.

        PROPOSED: Yes, this is reasonable. We should add "EGL_KHR_image_base
        affects the behavior of this extension" sections to the ES and VG
        extensions. Implementations can continue to export EGL_KHR_image if
        they are unable to support preserved image functionality.

    17. Do EGLImage Target creation extensions such as VG_KHR_EGL_image and
        GL_OES_EGL_image also need to be extended?

        UNRESOLVED: The problem here is that both these extensions
        explicitly state that pixel data becomes undefined when they
        reference an EGLImage to create an EGLImage target.

        One solution would be to allow this extension to do the defining on
        behalf of these extensions. For example, the VG_KHR_EGL_image
        extension on its own leaves the status of the pixel data undefined,
        but when VG_KHR_EGL_image is combined with this extension, then the
        status becomes defined (by this extension).

        When combined with the reasons given in Issue 1, this means it is
        possible to leave EGLImage Targe