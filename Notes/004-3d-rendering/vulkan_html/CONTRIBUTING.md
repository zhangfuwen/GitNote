<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#_vulkan_guide_contributing">Vulkan Guide Contributing</a>
<ul class="sectlevel1">
<li><a href="#_ways_to_contribute">1. Ways to Contribute</a></li>
<li><a href="#_images">2. Images</a></li>
<li><a href="#_markup">3. Markup</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/CONTRIBUTING.html
layout: default
---</p>
</div>
<h1 id="_vulkan_guide_contributing" class="sect0">Vulkan Guide Contributing</h1>
<div class="paragraph">
<p>While open for contributions from all, there are a few contribution rules in place.</p>
</div>
<div class="paragraph">
<p>The main design goal for the Vulkan Guide is to be &#8220;lean&#8221; and prevent any duplication of information. When possible the Vulkan Guide should guide a user to another resource via a hyperlink.</p>
</div>
<div class="sect1">
<h2 id="_ways_to_contribute">1. Ways to Contribute</h2>
<div class="sectionbody">
<div class="ulist">
<ul>
<li>
<p>Fixing a typo, grammar error, or other minor change</p>
<div class="ulist">
<ul>
<li>
<p>Please feel free to make a PR and it will hopefully get merged in quickly.</p>
</li>
</ul>
</div>
</li>
<li>
<p>Adding new content</p>
<div class="ulist">
<ul>
<li>
<p>If you think the guide needs another page, please raise an issue of what topic you feel is missing. This is not a requirement, but we hope to avoid people spending thier valuable time creating a PR for something that doesn&#8217;t quite belong in the guides.</p>
</li>
<li>
<p>If adding another link, clarification, or any other small blurb then a PR works. The addition of information needs to not be redundant and add meaningful value to the guide.</p>
</li>
</ul>
</div>
</li>
<li>
<p>Feel the guide is not accurately portraying a topic</p>
<div class="ulist">
<ul>
<li>
<p>There are a lot of ways to use Vulkan and the guide is aimed to be as objective as possible. This doesn&#8217;t mean that the current information on the Vulkan Guide might be slightly incorrect. Please raise an issue what you feel is incorrect and a solution to how you would improve it.</p>
</li>
</ul>
</div>
</li>
</ul>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_images">2. Images</h2>
<div class="sectionbody">
<div class="paragraph">
<p>All images must be no wider than 1080px. This is to force any large images to be resized to a more reasonable size.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_markup">3. Markup</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The Guide has been converted from Markdown to Asciidoc markup format. It is
possible to view the individual chapters (pages) as before, starting with
the repository root README.adoc, or to generate a single document containing
each chapter using 'make guide.html' with the 'asciidoctor' command-line
tool installed.</p>
</div>
<div class="paragraph">
<p>When adding new chapters, or adding cross-references to existing chapters,
it&#8217;s necessary to take these steps:</p>
</div>
<div class="olist arabic">
<ol class="arabic">
<li>
<p>For each chapter <code>chapters/page.adoc</code> which contains internal
cross-references, add this boilerplate at the top of the document:</p>
<div class="openblock">
<div class="content">
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-asciidoc" data-lang="asciidoc">ifndef::chapters[:chapters:]</code></pre>
</div>
</div>
<div class="paragraph">
<p>or, for chapters <code>chapters/extensions/page.adoc</code>, add this boilerplate:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-asciidoc" data-lang="asciidoc">ifndef::chapters[:chapters: ../]</code></pre>
</div>
</div>
</div>
</div>
</li>
<li>
<p>When creating an internal cross-reference link to <code>chapters/page.adoc</code>,
mark it up as follows:</p>
<div class="openblock">
<div class="content">
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-asciidoc" data-lang="asciidoc">This is a xref:{chapters}page.adoc#anchor[cross-reference]</code></pre>
</div>
</div>
<div class="paragraph">
<p>If cross-referencing <code>chapters/extensions/page.adoc</code>, mark it up as follows:</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-asciidoc" data-lang="asciidoc">This is a xref:{chapters}/extensions/page.adoc#anchor[cross-reference]</code></pre>
</div>
</div>
</div>
</div>
</li>
<li>
<p>The <code>anchor</code> is arbitrary, but is not optional. While asciidoctor will
generate anchor names automatically based on section titles, for link
stability it is best to use an explicit anchor on the title of the
target file:</p>
<div class="openblock">
<div class="content">
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-asciidoc" data-lang="asciidoc">ifndef::chapters[:chapters:]

[[anchor]]
= Target Page</code></pre>
</div>
</div>
</div>
</div>
</li>
</ol>
</div>
<div class="paragraph">
<p>Together, these steps allow cross-references to resolve correctly whether
viewing the Guide as a single document, or a set of related pages.</p>
</div>
</div>
</div>