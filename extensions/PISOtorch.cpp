
//#define PYBIND11_DETAILED_ERROR_MESSAGES

//#include "domain_structs.h"
#include "PISO_multiblock_cuda.h"
#include "grid_gen.h"
#include "resampling.h"
#include "eigenvalue.h"
#include "ortho_basis.h"
#include "matrix_vector_ops.h"

// File for the python bindings. The file has to have the same name as the python module for some build systems to work.

// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
/* needed for torch 1.13
namespace PYBIND11_NAMESPACE { namespace detail {
	// c10::optional binding is handled by torch (v1.13), but not its nullopt
	template <>
	struct type_caster<c10::nullopt_t> : public void_caster<c10::nullopt_t> {};
}}
*/

// 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
	//m.def("ConnectBlocks", static_cast<void (*)(std::shared_ptr<Block>, const dim_t, std::shared_ptr<Block>, const dim_t, const dim_t, const dim_t)>(&ConnectBlocks));
	//m.def("ConnectBlocks", static_cast<void (*)(std::shared_ptr<Block>, const std::string&, std::shared_ptr<Block>, const std::string&, const std::string&, const std::string&)>(&ConnectBlocks));

	py::class_<I4>(m, "Int4")
		.def(py::init(makeI4), py::arg("x")=0, py::arg("y")=0, py::arg("z")=0, py::arg("w")=0)
		.def_readonly("x", &I4::x)
		.def_readonly("y", &I4::y)
		.def_readonly("z", &I4::z)
		.def_readonly("w", &I4::w)
		.def("__getitem__", [](const I4 &self, const int32_t index){
			if(0<=index && index<4) return self.a[index];
			throw pybind11::index_error();
		})
		.def("__len__", [](const I4 &self){
			return 4;
		})
		.def("__str__", I4toString);
	
	py::enum_<BoundaryType>(m, "BoundaryType")
		.value("DIRICHLET", BoundaryType::DIRICHLET)
		.value("DIRICHLET_VARYING", BoundaryType::DIRICHLET_VARYING)
		.value("NEUMANN", BoundaryType::NEUMANN)
		.value("FIXED", BoundaryType::FIXED)
		.value("CONNECTED", BoundaryType::CONNECTED_GRID)
		.value("PERIODIC", BoundaryType::PERIODIC)
		.export_values();


	py::enum_<BoundaryConditionType>(m, "BoundaryConditionType")
		.value("DIRICHLET", BoundaryConditionType::DIRICHLET)
		.value("NEUMANN", BoundaryConditionType::NEUMANN)
		.export_values();
	
    py::class_<CSRmatrix, std::shared_ptr<CSRmatrix>>(m, "CSRmatrix")
        .def(py::init<torch::Tensor &, torch::Tensor &, torch::Tensor &>())
        //.def(py::init<index_t, index_t, torch::Dtype, torch::Device>())
        .def(py::init<index_t, index_t, py::object, torch::Device>())
        .def_readonly("value", &CSRmatrix::value)
		.def("setValue", &CSRmatrix::setValue)
		.def("CreateValue", &CSRmatrix::CreateValue)
        .def_readonly("index", &CSRmatrix::index)
        .def_readonly("row", &CSRmatrix::row)
		.def("getDevice", &CSRmatrix::getDevice)
		//.def("getDtype", &CSRmatrix::getDtype)
		.def("copy", &CSRmatrix::Copy)
		.def("clone", &CSRmatrix::Clone)
		.def("detach", &CSRmatrix::Detach, "detach value tensor gradient in-place.")
		.def("WithZeroValue", &CSRmatrix::WithZeroValue)
		.def("toType", &CSRmatrix::toTypePy)
		//.def("dtype", &CSRmatrix::dtype)
		//.def("scalar_type", &CSRmatrix::scalar_type)
		.def("getSize", &CSRmatrix::getSize)
		.def("getRows", &CSRmatrix::getRows)
		.def("__str__", &CSRmatrix::ToString)
		.def("IsTensorChanged", &CSRmatrix::IsTensorChanged);
    
	py::class_<Boundary, std::shared_ptr<Boundary>>(m, "Boundary")
		.def("getParentDomain", &Boundary::getParentDomain)
		.def_readonly("type", &Boundary::type)
		//.def("Copy", &Boundary::Copy)
		//.def("Clone", &Boundary::Clone)
		.def("__str__", &Boundary::ToString);
	
	py::class_<FixedBoundary, std::shared_ptr<FixedBoundary>, Boundary>(m, "FixedBoundary")
		//.def("Copy", &FixedBoundary::Copy)
		.def("DetachFwd", &FixedBoundary::DetachFwd)
		.def("DetachGrad", &FixedBoundary::DetachGrad)
		.def("Detach", &FixedBoundary::Detach)
		.def_readonly("velocity", &FixedBoundary::m_velocity)
		.def_readonly("velocityType", &FixedBoundary::m_velocityType)
		.def_readonly("isVelocityStatic", &FixedBoundary::m_velocityStatic)
		.def("setVelocity", &FixedBoundary::setVelocity)
		.def("setVelocityType", &FixedBoundary::setVelocityType)
		.def("CreateVelocity", &FixedBoundary::CreateVelocity)
		.def("getVelocity", &FixedBoundary::getVelocity,
			"Returns the velocity, broadcasted to varying shape, and optionally transformed to compuatational space.",
			py::arg("computational"))
		.def("getVelocityVarying", &FixedBoundary::getVelocityVarying,
			"Broadcast an existing static velocity to varying shape and returns it. The size argument can be used if no size has been set, otherwise it is ignored. Does not change the boundary's velocity.",
			py::arg("size")=nullopt)
		.def("makeVelocityVarying", &FixedBoundary::makeVelocityVarying,
			"Broadcast an existing static velocity to varying shape and sets it on the boundary. The size argument can be used if no size has been set, otherwise it is ignored.",
			py::arg("size")=nullopt)
		.def("GetFluxes", &FixedBoundary::GetFluxes,
			"Returns a tensor with the boundary-normal fluxes.")
		.def_readonly("passiveScalar", &FixedBoundary::m_passiveScalar)
		.def_readonly("passiveScalarTypes", &FixedBoundary::m_passiveScalarTypes)
		.def_readonly("_passiveScalarTypes_tensor", &FixedBoundary::m_passiveScalarTypes_tensor) // for debugging, TODO: remove
		.def("setPassiveScalar", &FixedBoundary::setPassiveScalar)
		.def("isPassiveScalarStatic", &FixedBoundary::isPassiveScalarStatic)
		.def("isPassiveScalarBoundaryTypeStatic", &FixedBoundary::isPassiveScalarBoundaryTypeStatic)
		.def("setPassiveScalarType", static_cast<void (FixedBoundary::*)(const BoundaryConditionType)>(&FixedBoundary::setPassiveScalarType))
		.def("setPassiveScalarType", static_cast<void (FixedBoundary::*)(const std::vector<BoundaryConditionType>)>(&FixedBoundary::setPassiveScalarType))
		.def("CreatePassiveScalar", &FixedBoundary::CreatePassiveScalar,
			py::arg("createStatic"))
		.def("hasPassiveScalar", &FixedBoundary::hasPassiveScalar)
		.def("getPassiveScalarChannels", &FixedBoundary::getPassiveScalarChannels)
		//.def("clearPassiveScalar", &FixedBoundary::clearPassiveScalar)
		.def_readonly("transform", &FixedBoundary::m_transform)
		.def("hasTransform", &FixedBoundary::hasTransform)
		.def("setTransform", &FixedBoundary::setTransform)
		.def("clearTransform", &FixedBoundary::clearTransform)
#ifdef WITH_GRAD
		.def_readonly("velocityGrad", &FixedBoundary::m_velocity_grad)
		.def("setVelocityGrad", &FixedBoundary::setVelocityGrad)
		.def("CreateVelocityGrad", &FixedBoundary::CreateVelocityGrad)
		//.def_readonly("pressureGrad", &FixedBoundary::pressure_grad)
		//.def("setPressureGrad", &FixedBoundary::setPressureGrad)
		//.def("CreatePressureGrad", &FixedBoundary::CreatePressureGrad)
		.def_readonly("passiveScalarGrad", &FixedBoundary::m_passiveScalar_grad)
		.def("setPassiveScalarGrad", &FixedBoundary::setPassiveScalarGrad)
		.def("CreatePassiveScalarGrad", &FixedBoundary::CreatePassiveScalarGrad)
#endif //WITH_GRAD
		.def("getSpatialDims", &FixedBoundary::getSpatialDims)
		.def("hasSize", &FixedBoundary::hasSize)
		.def("getSizes", &FixedBoundary::getSizes)
		.def("getStrides", &FixedBoundary::getStrides)
		.def("getDevice", &FixedBoundary::getDevice)
		//.def("getDtype", &FixedBoundary::getDtype)
		.def("__str__", &FixedBoundary::ToString);
	
	py::class_<StaticDirichletBoundary, std::shared_ptr<StaticDirichletBoundary>, Boundary>(m, "StaticDirichletBoundary")
		//.def(py::init<torch::Tensor &, torch::Tensor &, torch::Tensor &>())
		//.def("Copy", &StaticDirichletBoundary::Copy)
		.def_readonly("type", &StaticDirichletBoundary::type)
		.def_readonly("slip", &StaticDirichletBoundary::slip)
		.def_readonly("boundaryVelocity", &StaticDirichletBoundary::boundaryVelocity)
		.def_readonly("boundaryScalar", &StaticDirichletBoundary::boundaryScalar)
		.def("getSpatialDims", &StaticDirichletBoundary::getSpatialDims)
		.def("__str__", &StaticDirichletBoundary::ToString);
	
	py::class_<VaryingDirichletBoundary, std::shared_ptr<VaryingDirichletBoundary>, Boundary>(m, "VaryingDirichletBoundary")
		//.def(py::init<torch::Tensor &, torch::Tensor &, torch::Tensor &>())
		//.def("Copy", &VaryingDirichletBoundary::Copy)
		.def_readonly("type", &VaryingDirichletBoundary::type)
		.def_readonly("slip", &VaryingDirichletBoundary::slip)
		.def_readonly("boundaryVelocity", &VaryingDirichletBoundary::boundaryVelocity)
		.def("setVelocity", &VaryingDirichletBoundary::setVelocity)
		.def_readonly("boundaryScalar", &VaryingDirichletBoundary::boundaryScalar)
		.def("setPassiveScalar", &VaryingDirichletBoundary::setPassiveScalar)
		.def("setTransform", &VaryingDirichletBoundary::setTransform)
		.def_readonly("hasTransform", &VaryingDirichletBoundary::hasTransform)
		.def_readonly("transform", &VaryingDirichletBoundary::transform)
		.def("clearTransform", &VaryingDirichletBoundary::clearTransform)
#ifdef WITH_GRAD
		.def_readonly("boundaryVelocityGrad", &VaryingDirichletBoundary::boundaryVelocity_grad)
		.def("setVelocityGrad", &VaryingDirichletBoundary::setVelocityGrad)
		.def("CreateVelocityGrad", &VaryingDirichletBoundary::CreateVelocityGrad)
		.def_readonly("boundaryScalarGrad", &VaryingDirichletBoundary::boundaryScalar_grad)
		.def("setPassiveScalarGrad", &VaryingDirichletBoundary::setPassiveScalarGrad)
		.def("CreatePassiveScalarGrad", &VaryingDirichletBoundary::CreatePassiveScalarGrad)
#endif //WITH_GRAD
		.def("getSpatialDims", &VaryingDirichletBoundary::getSpatialDims)
		.def("getSizes", &VaryingDirichletBoundary::getSizes)
		.def("getStrides", &VaryingDirichletBoundary::getStrides)
		.def("__str__", &VaryingDirichletBoundary::ToString);
	
	py::class_<PeriodicBoundary, std::shared_ptr<PeriodicBoundary>, Boundary>(m, "PeriodicBoundary")
		//.def(py::init<>())
		.def_readonly("type", &PeriodicBoundary::type)
		.def("__str__", &PeriodicBoundary::ToString);
	
	py::class_<ConnectedBoundary, std::shared_ptr<ConnectedBoundary>, Boundary>(m, "ConnectedBoundary")
		//.def(py::init<std::shared_ptr<Block>, std::vector<dim_t> &>())
		.def_readonly("type", &ConnectedBoundary::type)
		.def("getConnectedBlock", &ConnectedBoundary::getConnectedBlock)
		.def_readonly("axes", &ConnectedBoundary::axes)
		/* .def_readonly("connectedFace", &ConnectedBoundary::connectedFace)
		.def_readonly("connectedAxis1", &ConnectedBoundary::connectedAxis1)
		.def_readonly("connectedAxis1", &ConnectedBoundary::connectedAxis2) */
		.def("__str__", &ConnectedBoundary::ToString);
	
	py::class_<Block, std::shared_ptr<Block>>(m, "Block")
		// Constructors are private, use Domain::CreateBlock to create blocks.
		//.def(py::init<torch::Tensor &, optional<torch::Tensor>, optional<torch::Tensor>, std::string &>())
		//.def(py::init<I4, std::string &, py::object, torch::Device>())
		//.def("Copy", &Block::Copy)
		.def("getParentDomain", &Block::getParentDomain)
		.def("DetachFwd", &Block::DetachFwd)
		.def("DetachGrad", &Block::DetachGrad)
		.def("Detach", &Block::Detach)
		.def_readonly("name", &Block::name)
		.def_readonly("velocity", &Block::velocity)
		.def("setVelocity", &Block::setVelocity)
		.def("getVelocity", &Block::getVelocity,
			"Returns the velocity, optionally transformed to compuatational space.",
			py::arg("computational"))
		.def("CreateVelocity", &Block::CreateVelocity)
		.def_readonly("velocitySource", &Block::velocitySource)
		.def_readonly("isVelocitySourceStatic", &Block::velocitySourceStatic)
		.def("hasVelocitySource", &Block::hasVelocitySource)
		.def("setVelocitySource", &Block::setVelocitySource)
		.def("CreateVelocitySource", &Block::CreateVelocitySource)
		.def("clearVelocitySource", &Block::clearVelocitySource)
		.def_readonly("viscosity", &Block::m_viscosity)
		.def("setViscosity", &Block::setViscosity)
		.def("hasViscosity", &Block::hasViscosity)
		.def("isViscosityStatic", &Block::isViscosityStatic)
		.def("clearViscosity", &Block::clearViscosity)
		.def_readonly("pressure", &Block::pressure)
		.def("setPressure", &Block::setPressure)
		.def("CreatePressure", &Block::CreatePressure)
		.def_readonly("passiveScalar", &Block::passiveScalar)
		.def("setPassiveScalar", &Block::setPassiveScalar)
		.def("CreatePassiveScalar", &Block::CreatePassiveScalar)
		//.def("clearPassiveScalar", &Block::clearPassiveScalar)
		.def("hasPassiveScalar", &Block::hasPassiveScalar)
		.def("getPassiveScalarChannels", &Block::getPassiveScalarChannels)
		.def_readonly("passiveScalarViscosity", &Block::m_passiveScalarViscosity)
		.def("hasPassiveScalarViscosity", &Block::hasPassiveScalarViscosity)
		// Coordinates and Transformations
		.def("setVertexCoordinates", &Block::setVertexCoordinates,
			"Set the cell vertex coordinates, automatically calculates cell and face transformations and updates existing StaticDirichletBoundaries.",
			py::arg("vertexCoordinates"))
		.def("hasVertexCoordinates", &Block::hasVertexCoordinates)
		.def("getCellCoordinates", &Block::getCellCoordinates)
		.def_readonly("vertexCoordinates", &Block::m_vertexCoordinates)
		.def("setTransform", &Block::setTransform,
			"Set cell transformation metrics, face transformations optional. Clears any previous transformations or coordinates.",
			py::arg("transform"), py::arg("faceTransform")=nullopt)
		.def_readonly("transform", &Block::m_transform)
		.def("hasTransform", &Block::hasTransform)
		.def("getCellSizes", &Block::getCellSizes, "Returns the block's cell sizes as NCDHW tensor (with C=1)")
		.def_readonly("faceTransform", &Block::m_faceTransform)
		.def("hasFaceTransform", &Block::hasFaceTransform)
		.def("clearCoordsTransforms", &Block::clearCoordsTransforms, "Clears any set coordinates or transformations.")
		.def("GetCoordinateOrientation", &Block::GetCoordinateOrientation,
			"Check if all transformed coordinate systems have the same orientation/handedness by checking the sign of the determinant."
			"Returns the sign if all have the same orientation, 0 otherwise."
			"Returns 1 if no transformations are set.")
#ifdef WITH_GRAD
		.def_readonly("velocityGrad", &Block::velocity_grad)
		.def("setVelocityGrad", &Block::setVelocityGrad)
		.def("CreateVelocityGrad", &Block::CreateVelocityGrad)
		.def_readonly("velocitySourceGrad", &Block::velocitySource_grad)
		.def("hasVelocitySourceGrad", &Block::hasVelocitySourceGrad)
		.def("setVelocitySourceGrad", &Block::setVelocitySourceGrad)
		.def("CreateVelocitySourceGrad", &Block::CreateVelocitySourceGrad, "only create grad if velocity source exists, clears it otherwise")
		.def("clearVelocitySourceGrad", &Block::clearVelocitySourceGrad)
		.def_readonly("viscosityGrad", &Block::m_viscosity_grad)
		.def("setViscosityGrad", &Block::setViscosityGrad)
		.def("CreateViscosityGrad", &Block::CreateViscosityGrad)
		.def("hasViscosityGrad", &Block::hasViscosityGrad)
		.def("clearViscosityGrad", &Block::clearViscosityGrad)
		.def_readonly("pressureGrad", &Block::pressure_grad)
		.def("setPressureGrad", &Block::setPressureGrad)
		.def("CreatePressureGrad", &Block::CreatePressureGrad)
		.def_readonly("passiveScalarGrad", &Block::passiveScalar_grad)
		.def("setPassiveScalarGrad", &Block::setPassiveScalarGrad)
		.def("CreatePassiveScalarGrad", &Block::CreatePassiveScalarGrad)
#endif //WITH_GRAD
		.def("IsTensorChanged", &Block::IsTensorChanged)
		.def("getMaxVelocity", &Block::getMaxVelocity, py::arg("withBounds")=true, py::arg("computational")=false)
		.def("getMaxVelocityMagnitude", &Block::getMaxVelocityMagnitude, py::arg("withBounds")=true, py::arg("computational")=false)
		.def("getDim", &Block::getDim)
		.def("getSpatialDims", &Block::getSpatialDims)
		// Boundaries
		.def("_setBoundary", static_cast<void (Block::*)(const index_t, std::shared_ptr<Boundary>)>(&Block::setBoundary),
			"Directly set a boundary to a specific face. Please use the boundary factories below to create consistent boundaries.",
			py::arg("faceIndex"), py::arg("boundary"))
		.def("_setBoundary", static_cast<void (Block::*)(const std::string&, std::shared_ptr<Boundary>)>(&Block::setBoundary),
			"Directly set a boundary to a specific face. Please use the boundary factories below to create consistent boundaries.",
			py::arg("faceString"), py::arg("boundary"))
		.def("getBoundary", static_cast<std::shared_ptr<Boundary> (Block::*)(const index_t) const>(&Block::getBoundary),
			py::arg("faceIndex"))
		.def("getBoundary", static_cast<std::shared_ptr<Boundary> (Block::*)(const std::string&) const>(&Block::getBoundary),
			py::arg("faceString"))
		.def("getBoundaries", &Block::getBoundaries)
		.def("hasPrescribedBoundary", &Block::hasPrescribedBoundary)
		.def("getFixedBoundaries", &Block::getFixedBoundaries,
			"Returns a list of pairs (boundary index, boundary) containing only the fixed boundaries of this block.")
		.def("isAllFixedBoundariesPassiveScalarTypeStatic", &Block::isAllFixedBoundariesPassiveScalarTypeStatic)
		.def("ConnectBlock", static_cast<void (Block::*)(const dim_t, std::shared_ptr<Block>, const dim_t, const dim_t, const dim_t)>(&Block::ConnectBlock),
			"Make a connection between this block and another."
			"directional face specification: [-x,+x,-y,+y,-z,+z] <-> [0,5]"
			"=> axis := face/2 ([x,y,z] <-> [0,2])"
			"=> direction := face%2, 0 is lower/negative side, 1 is upper/positive side"
			"	for 'axisIndex' the direction indicates if the connection is inverted (0 for same direction, 1 for inverted)"
			"faceIndex of the block is connected to otherFaceIndex of otherBlock."
			"for 2D and 3D, the remaining axes are also mapped:"
			"	faceIndex connects to otherFaceIndex"
			"	axis[(faceIndex / 2 + 1)%dims] is aligned to axis1Index. The connection is inverted if axis1Index%2==1."
			"	axis[(faceIndex / 2 + 2)%dims] is aligned to axis2Index. The connection is inverted if axis2Index%2==1.",
			py::arg("faceIndex"), py::arg("otherBlock"), py::arg("otherFaceIndex"), py::arg("axis1Index")=-1, py::arg("axis2Index")=-1)
		.def("ConnectBlock", static_cast<void (Block::*)(const std::string&, std::shared_ptr<Block>, const std::string&, const std::string&, const std::string&)>(&Block::ConnectBlock),
			"Make a connection between this block and another",
			py::arg("faceString"), py::arg("otherBlock"), py::arg("otherFaceString"), py::arg("axis1String")="", py::arg("axis2String")="")
		.def("MakePeriodic", static_cast<void (Block::*)(const index_t)>(&Block::MakePeriodic),
			"Make the block periodic along the given logical axis",
			py::arg("axisIndex"))
		.def("MakePeriodic", static_cast<void (Block::*)(const std::string&)>(&Block::MakePeriodic),
			"Make the block periodic along the given logical axis",
			py::arg("axisString"))
		.def("MakeAllPeriodic", &Block::MakeAllPeriodic,
			"Make the block perioic along all axes")
		.def("CloseBoundary_Old", static_cast<void (Block::*)(const index_t)>(&Block::CloseBoundary),
			"Create a DirichletBoundary with Dirichlet-0 velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.",
			py::arg("faceIndex"))
		.def("CloseBoundary_Old", static_cast<void (Block::*)(const std::string&)>(&Block::CloseBoundary),
			"Create a DirichletBoundary with Dirichlet-0 velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.",
			py::arg("faceString"))
		.def("CloseAllBoundaries", &Block::CloseAllBoundaries,
			"Create FixedBoundary with Dirichlet-0 velocity and scalar at all faces. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.")
		.def("CloseBoundary", static_cast<void (Block::*)(const index_t, optional<torch::Tensor>, optional<torch::Tensor>)>(&Block::CloseBoundary),
			"Create a FixedBoundary with Dirichlet velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.",
			py::arg("faceIndex"), py::arg("velocity")=nullopt, py::arg("passiveScalar")=nullopt)
		.def("CloseBoundary", static_cast<void (Block::*)(const std::string&, optional<torch::Tensor>, optional<torch::Tensor>)>(&Block::CloseBoundary),
			"Create a FixedBoundary with Dirichlet velocity and scalar at the specified face. Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well. The new boundaries will use existing face transformations.",
			py::arg("faceString"), py::arg("velocity")=nullopt, py::arg("passiveScalar")=nullopt)
		// misc
		.def("ComputeCSRSize", &Block::ComputeCSRSize)
		.def("getDevice", &Block::getDevice)
		//.def("getDtype", &Block::getDtype)
		.def("getSizes", &Block::getSizes)
		.def("getStrides", &Block::getStrides)
		//.def_readonly("numDims", &Block::numDims)
		.def_readonly("globalOffset", &Block::globalOffset)
		.def_readonly("csrOffset", &Block::csrOffset)
		.def("__str__", &Block::ToString);
	
	py::class_<Domain, std::shared_ptr<Domain>>(m, "Domain")
		//.def(py::init<std::string &>())
		.def(py::init<index_t, torch::Tensor&, std::string &, py::object, torch::Device, index_t, optional<torch::Tensor>>(),
			py::arg("spatialDims"), py::arg("viscosity"), py::arg("name"), py::arg("dtype"), py::arg("device"),
			py::arg("passiveScalarChannels")=1, py::arg("scalarViscosity")=nullopt)
		.def("Copy", &Domain::Copy,
			"Copy the domain structure (blocks, boundaries, connections) while keeping the same primary tensor references (velociity, pressure, etc.) as the original."
			"It is necessary to call PrepareSolve() on the copy to create the secondary tensors (matrices, RHS, results, gradients).",
			py::arg("newName")=nullopt)
		.def("Clone", &Domain::Clone,
			"Copy the domain structure (blocks, boundaries, connections) and clone (memory copy) the primary tensor references (velociity, pressure, etc.)."
			"It is necessary to call PrepareSolve() on the copy to create the secondary tensors (matrices, RHS, results, gradients).",
			py::arg("newName")=nullopt)
		.def("DetachFwd", &Domain::DetachFwd)
		.def("DetachGrad", &Domain::DetachGrad)
		.def("Detach", &Domain::Detach)
		.def("getMaxVelocity", &Domain::getMaxVelocity, py::arg("withBounds")=true, py::arg("computational")=false)
		.def("getMaxVelocityMagnitude", &Domain::getMaxVelocityMagnitude, py::arg("withBounds")=true, py::arg("computational")=false)
		.def("hasVertexCoordinates", &Domain::hasVertexCoordinates)
		.def("getVertexCoordinates", &Domain::getVertexCoordinates, "Returns a list of vertex coordnate tensors")
		.def("GetCoordinateOrientation", &Domain::GetCoordinateOrientation,
			"Check if all coordinate systems have the same orientation/handedness by checking the sign of the determinant."
			"Returns the sign if all have the same orientation, 0 otherwise.")
		.def("hasPrescribedBoundary", &Domain::hasPrescribedBoundary)
		.def("isAllFixedBoundariesPassiveScalarTypeStatic", &Domain::isAllFixedBoundariesPassiveScalarTypeStatic)
		.def("GetBoundaryFluxBalance", &Domain::GetGlobalFluxBalance)
		.def("CheckBoundaryFluxBalance", &Domain::CheckGlobalFluxBalance, py::arg("eps")=1e-5)
		// Blocks
		.def("CreateBlock", &Domain::CreateBlock,
			"Create a block on the domain, at least one field/tensor must be specified, the rest if initialzed with zero if required.",
			py::arg("velocity")=nullopt, py::arg("pressure")=nullopt, py::arg("passiveScalar")=nullopt, py::arg("vertexCoordinates")=nullopt, py::arg("name")="Block")
		.def("CreateBlockWithSize", &Domain::CreateBlockWithSize,
			"Create a zero-initialized block on the domain with the given spatial size.",
			py::arg("size"), py::arg("name")="Block")
		.def("AddBlock", &Domain::AddBlock)
		.def("getBlock", &Domain::getBlock)
		.def("getBlocks", &Domain::getBlocks)
		.def("PrepareSolve", &Domain::PrepareSolve)
		.def("UpdateDomainData", &Domain::UpdateDomain)
		.def("IsInitialized", &Domain::IsInitialized)
		.def("getNumBlocks", &Domain::getNumBlocks)
		.def("getSpatialDims", &Domain::getSpatialDims)
		.def("hasPassiveScalar", &Domain::hasPassiveScalar)
		.def("getPassiveScalarChannels", &Domain::getPassiveScalarChannels)
		.def("getTotalSize", &Domain::getTotalSize)
		.def("getDevice", &Domain::getDevice)
		.def("getDtype", &Domain::getPyDtype)
		//.def("getTensorOptions", &Domain::getTensorOptions)
		.def_readonly("viscosity", &Domain::viscosity)
		.def("setViscosity", &Domain::setViscosity)
		.def("hasBlockViscosity", &Domain::hasBlockViscosity, "Check if ANY block has viscosity set.")
		.def_readonly("passiveScalarViscosity", &Domain::passiveScalarViscosity)
		.def("hasPassiveScalarViscosity", &Domain::hasPassiveScalarViscosity)
		.def("isPassiveScalarViscosityStatic", &Domain::isPassiveScalarViscosityStatic)
		.def("setScalarViscosity", &Domain::setScalarViscosity)
		.def("clearScalarViscosity", &Domain::clearScalarViscosity)
		.def("hasPassiveScalarBlockViscosity", &Domain::hasPassiveScalarBlockViscosity, "Check if ANY block has passive scalar viscosity set.")
		.def("CreatePassiveScalarOnBlocks", &Domain::CreatePassiveScalarOnBlocks)
		.def("CreatePressureOnBlocks", &Domain::CreatePressureOnBlocks)
		.def("CreateVelocityOnBlocks", &Domain::CreateVelocityOnBlocks)
		.def_readonly("name", &Domain::name)
		.def_readonly("C", &Domain::C)
		.def_readonly("A", &Domain::A)
		.def("setA", &Domain::setA)
		.def("CreateA", &Domain::CreateA)
		.def_readonly("P", &Domain::P)
		.def_readonly("scalarRHS", &Domain::scalarRHS)
		.def("setScalarRHS", &Domain::setScalarRHS)
		.def("CreateScalarRHS", &Domain::CreateScalarRHS)
		.def_readonly("scalarResult", &Domain::scalarResult)
		.def("setScalarResult", &Domain::setScalarResult)
		.def("CreateScalarResult", &Domain::CreateScalarResult)
		.def_readonly("velocityRHS", &Domain::velocityRHS)
		.def("setVelocityRHS", &Domain::setVelocityRHS)
		.def("CreateVelocityRHS", &Domain::CreateVelocityRHS)
		.def_readonly("velocityResult", &Domain::velocityResult)
		.def("setVelocityResult", &Domain::setVelocityResult)
		.def("CreateVelocityResult", &Domain::CreateVelocityResult)
		.def_readonly("pressureRHS", &Domain::pressureRHS)
		.def("setPressureRHS", &Domain::setPressureRHS)
		.def("CreatePressureRHS", &Domain::CreatePressureRHS)
		.def_readonly("pressureRHSdiv", &Domain::pressureRHSdiv)
		.def("setPressureRHSdiv", &Domain::setPressureRHSdiv)
		.def("CreatePressureRHSdiv", &Domain::CreatePressureRHSdiv)
		.def_readonly("pressureResult", &Domain::pressureResult)
		.def("setPressureResult", &Domain::setPressureResult)
		.def("CreatePressureResult", &Domain::CreatePressureResult)
#ifdef WITH_GRAD
		.def("CreatePassiveScalarGradOnBlocks", &Domain::CreatePassiveScalarGradOnBlocks)
		.def("CreateVelocityGradOnBlocks", &Domain::CreateVelocityGradOnBlocks)
		.def("CreateVelocitySourceGradOnBlocks", &Domain::CreateVelocitySourceGradOnBlocks, "only create grad if velocity source exists, clears it otherwise")
		.def("CreatePressureGradOnBlocks", &Domain::CreatePressureGradOnBlocks)
		
		.def("CreatePassiveScalarGradOnBoundaries", &Domain::CreatePassiveScalarGradOnBoundaries)
		.def("CreateVelocityGradOnBoundaries", &Domain::CreateVelocityGradOnBoundaries)
		//.def("CreatePressureGradOnBoundaries", &Domain::CreatePressureGradOnBoundaries)
		
		.def_readonly("viscosityGrad", &Domain::viscosity_grad)
		.def("hasViscosityGrad", &Domain::hasViscosityGrad)
		.def("setViscosityGrad", &Domain::setViscosityGrad)
		.def("clearViscosityGrad", &Domain::clearViscosityGrad)
		.def("CreateViscosityGrad", &Domain::CreateViscosityGrad)
		
		.def_readonly("passiveScalarViscosityGrad", &Domain::passiveScalarViscosity_grad)
		.def("hasPassiveScalarViscosityGrad", &Domain::hasPassiveScalarViscosityGrad)
		.def("setPassiveScalarViscosityGrad", &Domain::setPassiveScalarViscosityGrad)
		.def("clearPassiveScalarViscosityGrad", &Domain::clearPassiveScalarViscosityGrad)
		.def("CreatePassiveScalarViscosityGrad", &Domain::CreatePassiveScalarViscosityGrad)
		
		.def_readonly("CGrad", &Domain::C_grad)
		.def_readonly("AGrad", &Domain::A_grad)
		.def("setAGrad", &Domain::setAGrad)
		.def("CreateAGrad", &Domain::CreateAGrad)
		.def_readonly("PGrad", &Domain::P_grad)
		
		.def_readonly("scalarRHSGrad", &Domain::scalarRHS_grad)
		.def("setScalarRHSGrad", &Domain::setScalarRHSGrad)
		.def("CreateScalarRHSGrad", &Domain::CreateScalarRHSGrad)
		.def_readonly("scalarResultGrad", &Domain::scalarResult_grad)
		.def("setScalarResultGrad", &Domain::setScalarResultGrad)
		.def("CreateScalarResultGrad", &Domain::CreateScalarResultGrad)
		
		.def_readonly("velocityRHSGrad", &Domain::velocityRHS_grad)
		.def("setVelocityRHSGrad", &Domain::setVelocityRHSGrad)
		.def("CreateVelocityRHSGrad", &Domain::CreateVelocityRHSGrad)
		.def_readonly("velocityResultGrad", &Domain::velocityResult_grad)
		.def("setVelocityResultGrad", &Domain::setVelocityResultGrad)
		.def("CreateVelocityResultGrad", &Domain::CreateVelocityResultGrad)
		
		.def_readonly("pressureRHSGrad", &Domain::pressureRHS_grad)
		.def("setPressureRHSGrad", &Domain::setPressureRHSGrad)
		.def("CreatePressureRHSGrad", &Domain::CreatePressureRHSGrad)
		.def_readonly("pressureRHSdivGrad", &Domain::pressureRHSdiv_grad)
		.def("setPressureRHSdivGrad", &Domain::setPressureRHSdivGrad)
		.def("CreatePressureRHSdivGrad", &Domain::CreatePressureRHSdivGrad)
		.def_readonly("pressureResultGrad", &Domain::pressureResult_grad)
		.def("setPressureResultGrad", &Domain::setPressureResultGrad)
		.def("CreatePressureResultGrad", &Domain::CreatePressureResultGrad)
#endif //WITH_GRAD
		.def("IsTensorChanged", &Domain::IsTensorChanged)
		.def_readonly("_packedCPU", &Domain::domainCPU)
		.def_readonly("_packedGPU", &Domain::domainGPU)
		.def("__str__", &Domain::ToString);
	
	/*
	py::enum_<NonOrthoFlags>(m, "NonOrthoFlags")
		.value("NONE", NonOrthoFlags::NONE)
		.value("ORTHOGONAL", NonOrthoFlags::ORTHOGONAL)
		.value("DIRECT_MATRIX", NonOrthoFlags::DIRECT_MATRIX)
		.value("DIRECT_RHS", NonOrthoFlags::DIRECT_RHS)
		.value("DIAGONAL_MATRIX", NonOrthoFlags::DIAGONAL_MATRIX)
		.value("DIAGONAL_RHS", NonOrthoFlags::DIAGONAL_RHS)
		//.value("ALL_RHS", NonOrthoMode::ALL_RHS)
		//.value("ALL_MATRIX", NonOrthoMode::ALL_MATRIX)
		.export_values();
		*/
	m.def("SetupAdvectionMatrix", &SetupAdvectionMatrixEulerImplicit,
		"Setup the matrix of the advection/diffusion system. Used for both velocity and passive scalars.",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoFlags"), py::arg("forPassiveScalar")=false, py::arg("passiveScalarChannel")=0);
	m.def("SetupAdvectionScalar", &SetupAdvectionScalarEulerImplicitRHS,
		"Setup the RHS for passive scalar advection/diffusion.",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoFlags"));
	m.def("SetupAdvectionVelocity", &SetupAdvectionVelocityEulerImplicitRHS,
		"Setup the RHS for velocity advection/diffusion.",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoFlags"), py::arg("applyPressureGradient")=false);
	//m.def("SetupAdvectionCombined", &SetupAdvectionEulerImplicitCombined, "PISO setup advection (CUDA)");
	m.def("SetupPressureCorrection", &SetupPressureCorrection,
		"Setup matrix, RHS, and div(RHS) of the pressure system.",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false, py::arg("timeStepNorm")=false);
	m.def("SetupPressureMatrix", &SetupPressureMatrix,
		"Setup only the matrix of the pressure system.",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false);
	m.def("SetupPressureRHS", &SetupPressureRHS,
		"Setup RHS and div(RHS) of the pressure system.",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false, py::arg("timeStepNorm")=false);
	m.def("SetupPressureRHSdiv", &SetupPressureRHSdiv,
		"Compute only div(RHS).",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false, py::arg("timeStepNorm")=false);
	m.def("CorrectVelocity", &CorrectVelocity,
		"Correct the velocity given by pressureRHSdiv with the gradient of the blocks' pressure fields.",
		py::arg("domain"), py::arg("timeStep"), py::arg("version")=0, py::arg("timeStepNorm")=false);
	m.def("ComputeVelocityDivergence", &ComputeVelocityDivergence, "velocity divergence (CUDA)");
	m.def("ComputePressureGradient", &ComputePressureGradient, "pressure gradient (CUDA)");
	m.def("ComputeSpatialVelocityGradients", &ComputeSpatialVelocityGradients,
		"Compute the spatial gradients of all velocity components of all blocks in the domain. Returns nested lists of tensors: [Blocks: [Components: NCDHW]]",
		py::arg("domain"));
	m.def("CopyScalarResultToBlocks", &CopyScalarResultToBlocks, "PISO copy scalar result (CUDA)");
	m.def("CopyScalarResultFromBlocks", &CopyScalarResultFromBlocks, "PISO copy scalar result (CUDA)");
	m.def("CopyPressureResultToBlocks", &CopyPressureResultToBlocks, "PISO copy pressure result (CUDA)");
	m.def("CopyPressureResultFromBlocks", &CopyPressureResultFromBlocks, "PISO copy pressure result (CUDA)");
	m.def("CopyVelocityResultToBlocks", &CopyVelocityResultToBlocks, "PISO copy velocity result (CUDA)");
	m.def("CopyVelocityResultFromBlocks", &CopyVelocityResultFromBlocks, "PISO copy velocity result back (CUDA)");
	
	m.def("SGSviscosityIncompressibleSmagorinsky", &SGSviscosityIncompressibleSmagorinsky,
		"Compute the additive viscosities for a Smagorinsky SGS scheme based on the velocity field. Returns a list of viscosity tensors, one per block.",
		py::arg("domain"), py::arg("coefficient"));
	
	py::enum_<ConvergenceCriterion>(m, "ConvergenceCriterion")
		.value("NORM2", ConvergenceCriterion::NORM2)
		.value("NORM2_NORMALIZED", ConvergenceCriterion::NORM2_NORMALIZED)
		.value("ABS_SUM", ConvergenceCriterion::ABS_SUM)
		.value("ABS_MEAN", ConvergenceCriterion::ABS_MEAN)
		.value("ABS_MAX", ConvergenceCriterion::ABS_MAX)
		.export_values();

	py::class_<LinearSolverResultInfo>(m, "LinearSolverResultInfo")
		//.def(py::init(makeI4), py::arg("x")=0, py::arg("y")=0, py::arg("z")=0, py::arg("w")=0)
		.def_readonly("finalResidual", &LinearSolverResultInfo::finalResidual)
		.def_readonly("usedIterations", &LinearSolverResultInfo::usedIterations)
		.def_readonly("converged", &LinearSolverResultInfo::converged)
		.def_readonly("isFiniteResidual", &LinearSolverResultInfo::isFiniteResidual)
		.def("__str__", LinearSolverResultInfoToString);
	m.def("SolveLinear", &SolveLinear, "Sparse linear solve on GPU (CUDA). With option for CG, BiCGStab, and preconditioning.",
		py::arg("A"), py::arg("RHS"), py::arg("x"), py::arg("maxIterations")=1000, py::arg("tolerance")=1e-8,
		py::arg("convergenceCriterion")=ConvergenceCriterion::NORM2_NORMALIZED,
		py::arg("useBiCG")=false, py::arg("matrixRankDeficient")=false, py::arg("residualResetSteps")=0, py::arg("transposeA")=false,
		py::arg("printResidual")=false, py::arg("returnBestResult")=false, py::arg("BiCGwithPreconditioner")=true);
	m.def("SparseOuterProduct", &SparseOuterProduct, "Outer product multiplied by sparsity pattern of result matrix. (CUDA)");
	m.def("TransformVectors", &TransformVectors, "Transform vectors with given transformation. (CUDA)");

#ifdef WITH_GRAD
	m.def("SetupAdvectionMatrixGrad", &SetupAdvectionMatrixEulerImplicit_GRAD,
		"PISO setup A matrix gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoFlags"), py::arg("forPassiveScalar")=false, py::arg("passiveScalarChannel")=0);
	m.def("SetupAdvectionScalarGrad", &SetupAdvectionScalarEulerImplicitRHS_GRAD,
		"PISO setup scalar advection gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoFlags"));
	m.def("SetupAdvectionVelocityGrad", &SetupAdvectionVelocityEulerImplicitRHS_GRAD,
	"PISO setup velocity advection gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoFlags"), py::arg("applyPressureGradient")=false);
	
	m.def("SetupPressureCorrectionGrad", &SetupPressureCorrection_GRAD, "PISO setup pressure gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false, py::arg("timeStepNorm")=false);
	m.def("SetupPressureMatrixGrad", &SetupPressureMatrix_GRAD, "PISO setup pressure gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false);
	m.def("SetupPressureRHSGrad", &SetupPressureRHS_GRAD, "PISO setup pressure gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false, py::arg("timeStepNorm")=false);
	m.def("SetupPressureRHSdivGrad", &SetupPressureRHSdiv_GRAD, "PISO setup pressure gradient (CUDA)",
		py::arg("domain"), py::arg("timeStep"), py::arg("nonOrthoMode")=0, py::arg("useFaceTransform")=false, py::arg("timeStepNorm")=false);
	m.def("CorrectVelocityGrad", &CorrectVelocity_GRAD, "PISO correct velocity gradient (CUDA)");
	
	m.def("CopyScalarResultGradFromBlocks", &CopyScalarResultGradFromBlocks, "PISO copy scalar result grad (CUDA)");
	m.def("CopyScalarResultGradToBlocks", &CopyScalarResultGradToBlocks, "PISO copy scalar result grad (CUDA)");
	m.def("CopyPressureResultGradFromBlocks", &CopyPressureResultGradFromBlocks, "PISO copy scalar result grad (CUDA)");
	m.def("CopyVelocityResultGradFromBlocks", &CopyVelocityResultGradFromBlocks, "PISO copy velocity result grad (CUDA)");
	m.def("CopyVelocityResultGradToBlocks", &CopyVelocityResultGradToBlocks, "PISO copy velocity result grad (CUDA)");

#endif //WITH_GRAD

	m.def("MakeGrid2DNonUniformScale", &MakeGrid2DNonUniformScale, "MakeGrid2DNonUniformScale (CUDA)");
	m.def("MakeGridNDNonUniformScaleNormalized", &MakeGridNDNonUniformScaleNormalized, "MakeGridNDNonUniformScaleNormalized (CUDA)");
	m.def("MakeGridNDExpScaleNormalized", &MakeGridNDExpScaleNormalized, "MakeGridNDExpScaleNormalized (CUDA)");
	m.def("MakeCoordsNDNonUniformScaleNormalized", &MakeCoordsNDNonUniformScaleNormalized, "MakeCoordsNDNonUniformScaleNormalized (CUDA)");
	m.def("CoordsToTransforms", &CoordsToTransforms, "Computes cell transformation metrics from cell vertex coordinates.");
	m.def("CoordsToFaceTransforms", &CoordsToFaceTransforms, "Computes cell face transformation metrics from cell vertex coordinates.");
	
	// vector-matrix operations
	m.def("matmul",
		static_cast<torch::Tensor (*)(const torch::Tensor&, const torch::Tensor&, const bool, const bool, const bool, const bool, const bool, const bool)>(&matmul),
		"Computes the product matrix/vector * matrix/vector for vectors or symmetrics matrices given in the channel dimension of NCDHW tensors."
		"Matrices are assumed to be in a flattend row-major format."
		"transpose and invert only apply if the quantity is a matrix.",
		py::arg("vectorMatrixA"), py::arg("vectorMatrixB"),
		py::arg("transposeA")=false, py::arg("invertA")=false,
		py::arg("transposeB")=false, py::arg("invertB")=false,
		py::arg("transposeOutput")=false, py::arg("invertOutput")=false);
#ifdef WITH_GRAD
	m.def("matmulGrad",
		static_cast<std::vector<torch::Tensor> (*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const bool, const bool, const bool, const bool, const bool, const bool)>(&matmulGrad),
		"Computes gradients of the product matrix/vector * matrix/vector for vectors or symmetrics matrices given in the channel dimension of NCDHW tensors."
		"Matrices are assumed to be in a flattend row-major format."
		"transpose and invert only apply if the quantity is a matrix. Gradients for inverted matrices are not computed.",
		py::arg("vectorMatrixA"), py::arg("vectorMatrixB"), py::arg("outputGrad"),
		py::arg("transposeA")=false, py::arg("invertA")=false,
		py::arg("transposeB")=false, py::arg("invertB")=false,
		py::arg("transposeOutput")=false, py::arg("invertOutput")=false);
#endif //WITH_GRAD
	m.def("VectorToDiagMatrix", &VectorToDiagMatrix, "Writes the vector in the channel dimension into the diagonal of a flat row-major matrix in the channel dimension.",
		py::arg("vectors"));
	m.def("InvertMatrix", &InvertMatrix, "Returns the inverse of a square matrix in the channel dimension.",
		py::arg("matrices"), py::arg("inPlace")=false);
	
	// eigen decomposition
	m.def("EigenDecomposition", &EigenDecomposition, "Eigen decomposition of symmetric matrices (CUDA). Returns tensors eigenvalues and eigenvectors.",
		py::arg("matrices"), py::arg("outputEigenvalues")=true, py::arg("outputEigenvectors")=true, py::arg("normalizeEigenvectors")=false);
	m.def("MakeBasisUnique", &MakeBasisUnique, "Makes a given set of orthogonal basis vectors (as colums in flat row-major matrices) unique (CUDA).",
		py::arg("basisMatrices"), py::arg("sortingVectors"), py::arg("inPlace")=false);
	
	
	// resampling
	py::enum_<BoundarySampling>(m, "BoundarySampling")
		.value("CONSTANT", BoundarySampling::CONSTANT)
		.value("CLAMP", BoundarySampling::CLAMP)
		.export_values();
	m.def("SampleTransformedGridGlobalToLocal", &SampleTransformedGridGlobalToLocal,
		"Sample from a globally transformed grid (single transformation matrix for all cells) to a locally transformed grid (individual cell coordinates)."
		"Each cell of the output grid gathers its value by interpolating cells of the input grid. This can lead to aliasing.",
		py::arg("globalData"), py::arg("globalTransform"), py::arg("localCoords"), py::arg("boundarySamplingMode"), py::arg("constantValue"));
	m.def("SampleTransformedGridLocalToGlobal", &SampleTransformedGridLocalToGlobal,
		"Sample from a locally transformed grid (individual cell coordinates) to a globally transformed grid (single transformation matrix for all cells)."
		"Each cell of the input grid scatters its value to cells of the output grid. The output grid is then normalized using the accumulated scattering weights."
		"This can lead to empty (zero) cells in the output."
		"Returns the output grid and scattering weights.",
		py::arg("localData"), py::arg("localCoords"), py::arg("globalTransform"), py::arg("globalShape"), py::arg("fillMaxSteps")=0);
	m.def("SampleTransformedGridLocalToGlobalMulti", &SampleTransformedGridLocalToGlobalMulti,
		"Version of SampleTransformedGridLocalToGlobal with multiple input grids.",
		py::arg("localData"), py::arg("localCoords"), py::arg("globalTransform"), py::arg("globalShape"), py::arg("fillMaxSteps")=0);

}