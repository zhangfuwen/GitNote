# KHR_no_error

Name

    KHR_no_error

Name Strings

    GL_KHR_no_error

Contributors

    Maurice Ribble
    Dominik Witczak
    Christophe Riccio
    Piers Daniell
    Jon Leech
    James Jones
    Daniel Kartch
    Steve Hill
    Jan-Harald Fredriksen

Contact

    Maurice Ribble (mribble 'at' qti.qualcomm.com)

Notice

    Copyright (c) 2012-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL and OpenGL ES Working Groups. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete.
    Approved by the Khronos Board of Promoters on May 8, 2015.

Version

    Version 6, February 25, 2015

Number

    ARB Extension #175
    OpenGL ES Extension #243

Dependencies

    Requires OpenGL ES 2.0 or OpenGL 2.0.

    Written against the OpenGL ES 3.1 specification.

    Interacts with EGL_KHR_create_context_no_error (or equivalent) extension.

    CONTEXT_FLAG_NO_ERROR_BIT_KHR is only supported if the OpenGL version has
    support for context flags (as defined in the OpenGL 4.5 core spec) or an
    extension supporting equivalent functionality is exposed.

Overview

    With this extension enabled any behavior that generates a GL error will
    have undefined behavior.  The reason this extension exists is performance
    can be increased and power usage decreased.  When this mode is used, a GL
    driver can have undefined behavior where it would have generated a GL error
    without this extension.  This could include application termination.  In
    general this extension should be used after you have verified all the GL
    errors are removed, and an application is not the kind that would check
    for GL errors and adjust behavior based on those errors.

New Procedures and Functions

    None

New Tokens

    CONTEXT_FLAG_NO_ERROR_BIT_KHR    0x00000008

Additions to the OpenGL ES Specification

    Add the following to the end of section 2.3.1 "Errors":

    If this context was created with the no error mode enabled then any place
    where the driver would have generated an error instead has undefined
    behavior.  This could include application termination.  All calls to
    GetError will return NO_ERROR or OUT_OF_MEMORY.  OUT_OF_MEMORY
    errors are a special case because they already allow for undefined behavior
    and are more difficult for application developers to predict than other
    errors.  This extension allows OUT_OF_MEMORY errors to be delayed, which
    can be useful for optimizing multithreaded drivers, but eventually the
    OUT_OF_MEMORY error should be reported if an implementation would have
    reported this error.  Since behavior of OUT_OF_MEMORY errors are undefined
    there is some implementation flexibility here.  However, this behavior may
    provide useful information on some implementations that do report
    OUT_OF_MEMORY without crashing.

    Add the following in the section that describes CONTEXT_FLAGS:

    If CONTEXT_FLAG_NO_ERROR_BIT_KHR is set in CONTEXT_FLAGS, then no error
    behavior is enabled for this context.

Errors

    None

New State

    None

Conformance Tests

    TBD

Issues

  (1) How does this extension interact with KHR_robustness and debug contexts?

  RESOLVED: The EGL/WGL/GLX layers should prevent these features from being
  enabled at the same time.  However, if they are somehow enabled at the same
  time this extension should be ignored.

  (2) Can we provide a guarantee bad usage of this API won't affect other apps
  that are running? This implies (but perhaps is not limited to) the risk of
  resource leaks (eg. If I use incorrect coordinates when executing load ops on
  images, am I guaranteed such ops will return the spec-guaranteed values or
  can I accidentally read someone else's memory?) and crashing other apps if I
  do horrendous stuff with the API.

  RESOLVED: GL already allows passing in invalid pointers in some calls that
  read or write to memory outside this apps process space.  Many drivers/OS
  models today offer protection from one process accessing memory from another
  process.  If you have that sort of protection before this extension then using
  this extension does not remove such protections.

  To put it another way this should not turn off kernel level or hardware level
  protections.

  (3) Should glGetError() always return NO_ERROR or have undefined results?

  RESOLVED: It should for all errors except OUT_OF_MEMORY.  For OUT_OF_MEMORY
  errors the error may be delayed to allow more optimization in multithreaded
  drivers, but if a driver would typically not crash and return OUT_OF_MEMORY
  then it should eventually return OUT_OF_MEMORY in this mode.  Since
  OUT_OF_MEMORY errors have undefined behavior this can't really be enforced.
  The reason OUT_OF_MEMORY errors aren't just converted to NO_ERROR like other
  errors is because some implementations don't crash on OUT_OF_MEMORY errors
  and apps can't easily predict when OUT_OF_MEMORY errors will happen so on
  these implementations apps might want to check for OUT_OF_MEMORY errors at
  certain points to see if things have gone very badly and then decide to do
  something else if they see an OUT_OF_MEMORY error.

  (4) Should we add something similar for CheckFramebufferStatus()?

  RESOLVED: NO.  It is unclear what the gain of this would be, and this is
  vendor specific behavior.  The same is true for ValidateProgram and
  ValidateProgramPipeline.  Also these features aren't as frequent as other GL
  calls so the gains from optimizing these would be much smaller.

  (5) Should there be a CONTEXT_FLAG for this behavior?

  RESOLVED: If CONTEXT_FLAGS are supported in the version of OpenGL being used
  or with an extension then yes.

Revision History

    Rev.    Date       Author     Changes
    ----  ------------ ---------  ----------------------------------------
      1   Jan 28, 2015 ribble     Initial version
      2   Jan 29, 2015 ribble     Added issues list
      3   Jan 30, 2015 ribble     Split into separate GL and EGL extensions
      4   Feb 18, 2015 ribble     Add special OOM error case and cleanup
      5   Feb 23, 2015 ribble     Add CONTEXT_FLAG issue
      6   Feb 25, 2015 ribble     Spec cleanup
