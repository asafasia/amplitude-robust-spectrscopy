# Repository instructions

## Experimental data

- The OPX1000 measurement data used by this project are stored in the sibling
  repository `/Users/asafsolonnikov/Developer/data_opx1000`.
- Treat that directory as read-only source data. Keep reproducible analysis and
  plotting code in this repository, and write derived figures under
  `figures/paper/` or `paper/figures/` as appropriate.
- Plotting scripts should accept `OPX1000_DATA_DIR` when practical, while using
  the sibling `data_opx1000` directory as the local default.
