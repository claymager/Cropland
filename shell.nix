with import <nixpkgs> {};
let dsPythonPackages = pythonPackages: with pythonPackages; [
  jupyter
  #Keras
  #opencv
  scikitimage
  numpy
  scikitlearn
  #dask
  pandas
  gdal
  pillow
  matplotlib
  flake8
];
in stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv {name = name; paths = buildInputs; };
  buildInputs = [
    (python3.withPackages dsPythonPackages)
    neovim
  ];
  shellHook = "fish";
}

