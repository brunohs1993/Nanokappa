To install pyembree in Windows 10, with Anaconda:

1. Open ANACONDA PROMPT (for some reason it seems not to work on standard command prompt).
2. Go to the desired environment folder (<anaconda folder>\envs\<environment>)
3. Run the following commands:

	cd Lib\site-packages
	git clone https://github.com/scopatz/pyembree.git
	cd pyembree
	conda install cython numpy
	conda install -c conda-forge embree
	set INCLUDE=%CONDA_PREFIX%\Library\include
	set LIB=%CONDA_PREFIX%\Library\lib
	python setup.py install --prefix=%CONDA_PREFIX%

#===== TROUBLESHOOTING ====#

When executed setup.py, you may find this error:
	fatal error C1083: Cannot open include file: 'io.h': No such file or directory.

Solution:
	Download visual studio build tools and install:

	- Visual C++ Build tools core features.
	- VC++ 2017 v141 toolset (x86,x64)
	- Visual C++ 2017 Redistributable Update
	- Windows 10 SDK (10.0.16299.0) for Desktop C++

	link: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

------------------------------

Other possible error:
	fatal error LNK1158: cannot run ‘rc.exe’
Stack Overflow solution:
	https://stackoverflow.com/questions/14372706/visual-studio-cant-build-due-to-rc-exe

-------------------------------
If new error, write solution here