{
  description = "Pinned Gufo source-extension package";
  inputs.gufo.url = "github:gufo-org/gufo/ae7ef25f350650a94be2f798706895fa0c8aa80f";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
