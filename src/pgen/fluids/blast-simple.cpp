//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <cmath>

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (restart) return;

  //Define the radius of the Blast:
  Real R = pin->GetReal("problem", "R");
  // values for neutrals (hydro fluid)
  Real p_amb   = pin->GetReal("problem", "p_amb");
  Real d_amb   = pin->GetReal("problem", "d_amb");
  // values for ions (hydro fluid)
  Real p_in   = pin->GetReal("problem", "p_in");
  Real d_in   = pin->GetReal("problem", "d_in");

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    auto &w0_ = pmbp->phydro->w0;
    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

    par_for("pgen_blast1",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

      Real den = d_amb;
      Real pres = p_amb;

      if (rad < R) {
        den = d_in;
        pres = p_in;
      }
      
      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVY,k,j,i) = 0.0;
      w0_(m,IVZ,k,j,i) = 0.0;
      w0_(m,IEN,k,j,i) = pres/gm1;
    });

  // convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);  
  }  
}

