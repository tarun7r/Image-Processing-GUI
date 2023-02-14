<h1>Image Processing Assignment - Basic Image Editor</h1>
	<p>This project aims to develop a basic image editor with a Graphical User Interface (GUI) that can handle color and grayscale images. The editor provides various image manipulation options, including equalization of histograms, gamma correction, log transformation, blurring, sharpening, and more. The editor also has an undo and redo feature and allows the user to save the current image.</p>
	<h2>Features</h2>
	<ul>
		<li>Image display area</li>
		<li>Image load button that opens a file selector.</li>
		<li>Capability to handle color and grayscale images. Color images are converted to HSI/HSV or Lab, and only the I/V/L channel is manipulated.</li>
		<li>Equalize histogram</li>
		<li>Gamma correct (ask for input gamma upon pressing the button)</li>
		<li>Log transform</li>
		<li>Blur with a mechanism to control the extent of blurring</li>
		<li>Sharpening with a mechanism to control the extent of sharpening</li>
		<li>Undo all changes (revert to original image)</li>
		<li>Save current image button</li>
		<li>Additional feature (surprise!)</li>
	</ul>
	<h2>Team Features (Two-Person Teams Only)</h2>
	<ul>
		<li>Compute 2-D DFT and display magnitude and phase</li>
		<li>Load a frequency magnitude mask</li>
		<li>Compute and display modified image using the mask</li>
	</ul>
  <h2> Usage </h2>
  <div class="highlight highlight-source-shell">
    <pre><span class="pl-c">$</span> git clone https://https://github.com/tarun7r/Image-Processing-GUI.git
<span class="pl-c">$</span> cd Image-Processing-GUI
<span class="pl-c">$</span> python gui.py</pre>
</div>
<h2>Conclusion</h2>
The image editor developed in this project is a basic tool that can handle color and grayscale images and provides a variety of manipulation options. Further improvements could include additional image processing options and an improved user interface.
