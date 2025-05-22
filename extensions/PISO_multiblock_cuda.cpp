#include <torch/extension.h>
#include <iostream>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_HOST(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a host tensor")
#define CHECK_HOST_SCALAR_FLOAT(x) CHECK_HOST(x); TORCH_CHECK(x.dim()==1 && x.size(0)==1, #x " must have shape [1]."); TORCH_CHECK(x.dtype()==torch::kFloat32, #x " must be a float tensor.")
#define CHECK_HOST_SCALAR_INT(x) CHECK_HOST(x); TORCH_CHECK(x.dim()==1 && x.size(0)==1, #x " must have shape [1]."); TORCH_CHECK(x.dtype()==torch::kInt32, #x " must be an integer tensor.")


std::vector<torch::Tensor> PISO_build_A_matrix_cuda_forward(torch::Tensor velocity, torch::Tensor boundaries);
std::vector<torch::Tensor> PISO_advect_cuda_forward(torch::Tensor data, torch::Tensor velocity, torch::Tensor boundaries);
std::vector<torch::Tensor> PISO_step_cuda_forward(torch::Tensor data, torch::Tensor velocity, torch::Tensor pressureGuess,
	torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU,
	torch::Tensor stepsCPU, torch::Tensor correctorStepsCPU, torch::Tensor timeStepCPU, torch::Tensor epsCPU,
	const int spatial_dimensions);

std::vector<torch::Tensor> SetupAdvectionMatrixEulerImplicite(torch::Tensor velocity,
	torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU);
std::vector<torch::Tensor> SetupAdvectionScalarEulerImpliciteRHS(torch::Tensor scalarData,
	torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU);
std::vector<torch::Tensor> SolveAdvectionScalarEulerImplicite(torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex, torch::Tensor dataAdvectionRHS,
	torch::Tensor epsCPU);
std::vector<torch::Tensor> SetupAdvectionVelocityEulerImpliciteRHS(torch::Tensor velocity, torch::Tensor pressureGuess,
	torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU);
std::vector<torch::Tensor> SolveAdvectionVelocityEulerImplicite(torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex, torch::Tensor velocityAdvectionRHS,
	torch::Tensor epsCPU);
std::vector<torch::Tensor> SetupPressurePISO(torch::Tensor velocity, torch::Tensor velocityPrediction,
	torch::Tensor matrixDiagonal, torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex,
	torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU);
std::vector<torch::Tensor> SolvePressurePISO(torch::Tensor pMatrixValue, torch::Tensor pMatrixRow, torch::Tensor pMatrixIndex, torch::Tensor pressureRHSdiv,
	torch::Tensor epsCPU);
std::vector<torch::Tensor> CorrectVelocityPISO(torch::Tensor matrixDiagonal, torch::Tensor pressureRHS, torch::Tensor pressureGuess,
	torch::Tensor timeStepCPU, torch::Tensor boundariesCPU, torch::Tensor boundaryValuesCPU);
/*
* velocity format: colocated/centered NDHWC
* pressure format: scalar grid
*
* boundaries: 1D int, [x+,x-,y+,y-,z+,z-], 0=open, 1=closed, 2=periodic
*/

std::vector<torch::Tensor> PISO_build_A_matrix_forward(torch::Tensor velocity, torch::Tensor boundaries){
	CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.size(0)==1, "batches are not yet supported.")
	const int spatial_dimensions = velocity.dim()-2;
	TORCH_CHECK(spatial_dimensions==3, "only 3D is supported.");
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==6, "boundaries must have shape [6].")
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.")
	
	return PISO_build_A_matrix_cuda_forward(velocity, boundaries);
}

std::vector<torch::Tensor> PISO_advect_forward(torch::Tensor data, torch::Tensor velocity, torch::Tensor boundaries){
	CHECK_INPUT_CUDA(data);
	CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.dim()==data.dim(), "dimensions must match")
	TORCH_CHECK(velocity.size(0)==1, "batches are not yet supported.")
	const int spatial_dimensions = velocity.dim()-2;
	//TORCH_CHECK(0<spatial_dimensions && spatial_dimensions<4, "must have 1, 2, or 3 spatial dimensions")
	TORCH_CHECK(spatial_dimensions==3, "only 3D is supported.");
	TORCH_CHECK(velocity.size(0)==data.size(0), "batch size must match")
	TORCH_CHECK(velocity.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions")
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==data.size(dim), "spatial dimension must match")
	}
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==6, "boundaries must have shape [6].")
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.")
	
	return PISO_advect_cuda_forward(data, velocity, boundaries);
}

//TODO: timestep, steps

std::vector<torch::Tensor> PISO_step_forward(torch::Tensor data, torch::Tensor velocity, torch::Tensor pressureGuess, torch::Tensor boundaries, torch::Tensor boundaryValues,
		torch::Tensor steps, torch::Tensor correctorSteps, torch::Tensor timeStep, torch::Tensor eps){
	CHECK_INPUT_CUDA(data);
	CHECK_INPUT_CUDA(velocity);
	CHECK_INPUT_CUDA(pressureGuess);
	TORCH_CHECK(velocity.dim()==data.dim(), "dimensions must match")
	TORCH_CHECK(velocity.dim()==pressureGuess.dim(), "dimensions must match")
	TORCH_CHECK(velocity.size(0)==1, "batches are not yet supported.")
	const int spatial_dimensions = velocity.dim()-2;
	//TORCH_CHECK(0<spatial_dimensions && spatial_dimensions<4, "must have 1, 2, or 3 spatial dimensions")
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(velocity.size(0)==data.size(0), "batch size must match")
	TORCH_CHECK(velocity.size(0)==pressureGuess.size(0), "batch size must match")
	TORCH_CHECK(velocity.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions")
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==data.size(dim), "spatial dimension must match")
		TORCH_CHECK(velocity.size(dim)==pressureGuess.size(dim), "spatial dimension must match")
	}
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==spatial_dimensions*2, "boundaries must have shape [dims*2].")
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.")
	CHECK_HOST(boundaryValues);
	TORCH_CHECK(boundaryValues.dim()==1 && boundaryValues.size(0)==spatial_dimensions*2, "boundary values must have shape [dims*2].")
	TORCH_CHECK(boundaryValues.dtype()==torch::kFloat32, "boundary values must be a float tensor.")
	
	CHECK_HOST(steps)
	TORCH_CHECK(steps.dim()==1 && steps.size(0)==1, "boundaries must have shape [1].")
	TORCH_CHECK(steps.dtype()==torch::kInt32, "boundaries must be an integer tensor.")
	
	CHECK_HOST(correctorSteps)
	TORCH_CHECK(correctorSteps.dim()==1 && correctorSteps.size(0)==1, "boundaries must have shape [1].")
	TORCH_CHECK(correctorSteps.dtype()==torch::kInt32, "boundaries must be an integer tensor.")
	
	CHECK_HOST(timeStep)
	TORCH_CHECK(timeStep.dim()==1 && timeStep.size(0)==1, "boundaries must have shape [1].")
	TORCH_CHECK(timeStep.dtype()==velocity.dtype(), "boundaries must be an float tensor.")
	
	CHECK_HOST(eps)
	TORCH_CHECK(eps.dim()==1 && eps.size(0)==1, "eps must have shape [1].")
	TORCH_CHECK(eps.dtype()==velocity.dtype(), "eps must be an float tensor.")
	
	return PISO_step_cuda_forward(data, velocity, pressureGuess, boundaries, boundaryValues, steps, correctorSteps, timeStep, eps, spatial_dimensions);
}




std::vector<torch::Tensor> Py_SetupAdvectionMatrixEulerImplicite(torch::Tensor velocity,
		torch::Tensor timeStep, torch::Tensor boundaries, torch::Tensor boundaryValues){
	CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = velocity.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(velocity.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions");
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==spatial_dimensions*2, "boundaries must have shape [dims*2].");
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.");
	CHECK_HOST(boundaryValues);
	TORCH_CHECK(boundaryValues.dim()==1 && boundaryValues.size(0)==spatial_dimensions*2, "boundary values must have shape [dims*2].");
	TORCH_CHECK(boundaryValues.dtype()==torch::kFloat32, "boundary values must be a float tensor.");
	
	CHECK_HOST(timeStep)
	TORCH_CHECK(timeStep.dim()==1 && timeStep.size(0)==1, "boundaries must have shape [1].");
	TORCH_CHECK(timeStep.dtype()==velocity.dtype(), "boundaries must be an float tensor.");

	return SetupAdvectionMatrixEulerImplicite(velocity, timeStep, boundaries, boundaryValues);
}
std::vector<torch::Tensor> Py_SetupAdvectionScalarEulerImpliciteRHS(torch::Tensor scalarData,
		torch::Tensor timeStep, torch::Tensor boundaries, torch::Tensor boundaryValues){
	CHECK_INPUT_CUDA(scalarData);
	TORCH_CHECK(scalarData.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = scalarData.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(scalarData.size(1)==1, "channels must be scalar (1)");
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==spatial_dimensions*2, "boundaries must have shape [dims*2].");
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.");
	CHECK_HOST(boundaryValues);
	TORCH_CHECK(boundaryValues.dim()==1 && boundaryValues.size(0)==spatial_dimensions*2, "boundary values must have shape [dims*2].");
	TORCH_CHECK(boundaryValues.dtype()==torch::kFloat32, "boundary values must be a float tensor.");
	
	CHECK_HOST(timeStep)
	TORCH_CHECK(timeStep.dim()==1 && timeStep.size(0)==1, "boundaries must have shape [1].");
	TORCH_CHECK(timeStep.dtype()==torch::kFloat32, "boundaries must be an float tensor.");

	return SetupAdvectionScalarEulerImpliciteRHS(scalarData, timeStep, boundaries, boundaryValues);
}
std::vector<torch::Tensor> Py_SolveAdvectionScalarEulerImplicite(torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex, torch::Tensor dataAdvectionRHS,
		torch::Tensor eps){
	CHECK_INPUT_CUDA(csrMatrixValue);
	CHECK_INPUT_CUDA(csrMatrixRow);
	CHECK_INPUT_CUDA(csrMatrixIndex);
	CHECK_INPUT_CUDA(dataAdvectionRHS);
	TORCH_CHECK(dataAdvectionRHS.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = dataAdvectionRHS.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(dataAdvectionRHS.size(1)==1, "channels must be scalar (1)");
	
	CHECK_HOST(eps)
	TORCH_CHECK(eps.dim()==1 && eps.size(0)==1, "eps must have shape [1].")
	TORCH_CHECK(eps.dtype()==torch::kFloat32, "eps must be an float tensor.")

	return SolveAdvectionScalarEulerImplicite(csrMatrixValue, csrMatrixRow, csrMatrixIndex, dataAdvectionRHS, eps);
}
std::vector<torch::Tensor> Py_SetupAdvectionVelocityEulerImpliciteRHS(torch::Tensor velocity, torch::Tensor pressureGuess,
		torch::Tensor timeStep, torch::Tensor boundaries, torch::Tensor boundaryValues){
	CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = velocity.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(velocity.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions");

	CHECK_INPUT_CUDA(pressureGuess);
	TORCH_CHECK(pressureGuess.dim()==velocity.dim(), "dimensions must match");
	TORCH_CHECK(pressureGuess.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(pressureGuess.size(1)==1, "pressure channels must be scalar (1)");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==pressureGuess.size(dim), "spatial dimension must match")
	}
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==spatial_dimensions*2, "boundaries must have shape [dims*2].");
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.");
	CHECK_HOST(boundaryValues);
	TORCH_CHECK(boundaryValues.dim()==1 && boundaryValues.size(0)==spatial_dimensions*2, "boundary values must have shape [dims*2].");
	TORCH_CHECK(boundaryValues.dtype()==torch::kFloat32, "boundary values must be a float tensor.");
	
	CHECK_HOST(timeStep)
	TORCH_CHECK(timeStep.dim()==1 && timeStep.size(0)==1, "boundaries must have shape [1].");
	TORCH_CHECK(timeStep.dtype()==torch::kFloat32, "boundaries must be an float tensor.");

	return SetupAdvectionVelocityEulerImpliciteRHS(velocity, pressureGuess, timeStep, boundaries, boundaryValues);
}
std::vector<torch::Tensor> Py_SolveAdvectionVelocityEulerImplicite(torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex, torch::Tensor velocityAdvectionRHS,
	torch::Tensor eps){
	CHECK_INPUT_CUDA(csrMatrixValue);
	CHECK_INPUT_CUDA(csrMatrixRow);
	CHECK_INPUT_CUDA(csrMatrixIndex);
	CHECK_INPUT_CUDA(velocityAdvectionRHS);
	TORCH_CHECK(velocityAdvectionRHS.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = velocityAdvectionRHS.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(velocityAdvectionRHS.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions");
	
	CHECK_HOST(eps)
	TORCH_CHECK(eps.dim()==1 && eps.size(0)==1, "eps must have shape [1].")
	TORCH_CHECK(eps.dtype()==torch::kFloat32, "eps must be an float tensor.")

	return SolveAdvectionVelocityEulerImplicite(csrMatrixValue, csrMatrixRow, csrMatrixIndex, velocityAdvectionRHS, eps);
}
std::vector<torch::Tensor> Py_SetupPressurePISO(torch::Tensor velocity, torch::Tensor velocityPrediction,
		torch::Tensor matrixDiagonal, torch::Tensor csrMatrixValue, torch::Tensor csrMatrixRow, torch::Tensor csrMatrixIndex,
		torch::Tensor timeStep, torch::Tensor boundaries, torch::Tensor boundaryValues){
	
	CHECK_INPUT_CUDA(matrixDiagonal);
	CHECK_INPUT_CUDA(csrMatrixValue);
	CHECK_INPUT_CUDA(csrMatrixRow);
	CHECK_INPUT_CUDA(csrMatrixIndex);
	CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = velocity.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(velocity.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions");
	
	CHECK_INPUT_CUDA(velocityPrediction);
	TORCH_CHECK(velocityPrediction.dim()==velocity.dim(), "dimensions must match");
	TORCH_CHECK(velocityPrediction.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(velocityPrediction.size(1)==spatial_dimensions, "velocityP channels must match spatial dimensions");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==velocityPrediction.size(dim), "spatial dimension must match");
	}
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==spatial_dimensions*2, "boundaries must have shape [dims*2].");
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.");
	CHECK_HOST(boundaryValues);
	TORCH_CHECK(boundaryValues.dim()==1 && boundaryValues.size(0)==spatial_dimensions*2, "boundary values must have shape [dims*2].");
	TORCH_CHECK(boundaryValues.dtype()==torch::kFloat32, "boundary values must be a float tensor.");
	
	CHECK_HOST(timeStep)
	TORCH_CHECK(timeStep.dim()==1 && timeStep.size(0)==1, "boundaries must have shape [1].");
	TORCH_CHECK(timeStep.dtype()==torch::kFloat32, "boundaries must be an float tensor.");

	return SetupPressurePISO(velocity, velocityPrediction,
		matrixDiagonal, csrMatrixValue, csrMatrixRow, csrMatrixIndex,
		timeStep, boundaries, boundaryValues);
}
std::vector<torch::Tensor> Py_SolvePressurePISO(torch::Tensor pMatrixValue, torch::Tensor pMatrixRow, torch::Tensor pMatrixIndex, torch::Tensor pressureRHSdiv,
		torch::Tensor eps){
	CHECK_INPUT_CUDA(pMatrixValue);
	CHECK_INPUT_CUDA(pMatrixRow);
	CHECK_INPUT_CUDA(pMatrixIndex);
	CHECK_INPUT_CUDA(pressureRHSdiv);
	TORCH_CHECK(pressureRHSdiv.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = pressureRHSdiv.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	//TORCH_CHECK(pressureRHSdiv.size(1)==1, "channels must be scalar (1)");
	
	CHECK_HOST(eps)
	TORCH_CHECK(eps.dim()==1 && eps.size(0)==1, "eps must have shape [1].")
	TORCH_CHECK(eps.dtype()==torch::kFloat32, "eps must be an float tensor.")

	return SolvePressurePISO(pMatrixValue, pMatrixRow, pMatrixIndex, pressureRHSdiv, eps);
}
std::vector<torch::Tensor> Py_CorrectVelocityPISO(torch::Tensor matrixDiagonal, torch::Tensor pressureRHS, torch::Tensor pressureGuess,
		torch::Tensor timeStep, torch::Tensor boundaries, torch::Tensor boundaryValues){
	
	CHECK_INPUT_CUDA(matrixDiagonal);
	CHECK_INPUT_CUDA(pressureRHS);
	TORCH_CHECK(pressureRHS.size(0)==1, "batches are not yet supported.");
	const int spatial_dimensions = pressureRHS.dim()-2;
	TORCH_CHECK(spatial_dimensions>0 && spatial_dimensions<4, "only 1D, 2D, and 3D is supported.");
	TORCH_CHECK(pressureRHS.size(1)==spatial_dimensions, "velocity channels must match spatial dimensions");
	
	CHECK_INPUT_CUDA(pressureGuess);
	TORCH_CHECK(pressureGuess.dim()==pressureRHS.dim(), "dimensions must match");
	TORCH_CHECK(pressureGuess.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(pressureGuess.size(1)==1, "pressure guess channels must be 1");
	for(int dim=2;dim<pressureRHS.dim();++dim){
		TORCH_CHECK(pressureRHS.size(dim)==pressureGuess.size(dim), "spatial dimension must match");
	}
	
	CHECK_HOST(boundaries);
	TORCH_CHECK(boundaries.dim()==1 && boundaries.size(0)==spatial_dimensions*2, "boundaries must have shape [dims*2].");
	TORCH_CHECK(boundaries.dtype()==torch::kInt32, "boundaries must be an integer tensor.");
	CHECK_HOST(boundaryValues);
	TORCH_CHECK(boundaryValues.dim()==1 && boundaryValues.size(0)==spatial_dimensions*2, "boundary values must have shape [dims*2].");
	TORCH_CHECK(boundaryValues.dtype()==torch::kFloat32, "boundary values must be a float tensor.");
	
	CHECK_HOST(timeStep)
	TORCH_CHECK(timeStep.dim()==1 && timeStep.size(0)==1, "boundaries must have shape [1].");
	TORCH_CHECK(timeStep.dtype()==torch::kFloat32, "boundaries must be an float tensor.");

	return CorrectVelocityPISO(matrixDiagonal, pressureRHS, pressureGuess,
		timeStep, boundaries, boundaryValues);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buildA", &PISO_build_A_matrix_forward, "PISO a matrix forward (CUDA)");
  m.def("advect", &PISO_advect_forward, "PISO advection forward (CUDA)");
  m.def("PISOstep", &PISO_step_forward, "PISO step forward (CUDA)");

  m.def("PISOsetupAmatrix", &Py_SetupAdvectionMatrixEulerImplicite, "PISO setup A matrix (CUDA)");
  m.def("PISOsetupScalarAdvection", &Py_SetupAdvectionScalarEulerImpliciteRHS, "PISO setup scalar advection (CUDA)");
  m.def("PISOsolveScalarAdvection", &Py_SolveAdvectionScalarEulerImplicite, "PISO solve scalar advection (CUDA)");
  m.def("PISOsetupVelocityAdvection", &Py_SetupAdvectionVelocityEulerImpliciteRHS, "PISO setup velocity advection (CUDA)");
  m.def("PISOsolveVelocityAdvection", &Py_SolveAdvectionVelocityEulerImplicite, "PISO solve velocity advection (CUDA)");
  m.def("PISOsetupPressure", &Py_SetupPressurePISO, "PISO setup pressure (CUDA)");
  m.def("PISOsolvePressure", &Py_SolvePressurePISO, "PISO solve pressure(CUDA)");
  m.def("PISOcorretVelocity", &Py_CorrectVelocityPISO, "PISO correct velocity(CUDA)");
}