git clone git://github.com/huyng/bashmarks.git
cd bashmarks;
make install
cd ..
rm -rf bashmarks

echo "source ~/.local/bin/bashmarks.sh" >> ~/.bashrc