#pragma once

#ifndef _INCLUDE_DOMAIN_STRUCTS
#define _INCLUDE_DOMAIN_STRUCTS

#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
//#include <optional>
#include <domain_structs_gpu.h>
//#include <include/tl/optional.hpp> trying c10::optional first

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_HOST(x) TORCH_CHECK(!x.device().is_cuda(), #x " must be a host tensor")
#define CHECK_INPUT_HOST(x) CHECK_HOST(x); CHECK_CONTIGUOUS(x)

#define GET_DATA_PRT(name, m_name) \
template <typename scalar_t> \
scalar_t* get##name##DataPtr() const { \
	return m_name.data_ptr<scalar_t>(); \
}

#define GET_OPTIONAL_DATA_PRT(name, m_name) \
template <typename scalar_t> \
scalar_t* get##name##DataPtr() const { \
	return m_name.has_value() ? m_name.value().data_ptr<scalar_t>() : nullptr; \
}

//#define DTOR_MSG

template <typename T>
using optional = c10::optional<T>;
const auto nullopt = c10::nullopt;

const torch::Dtype torch_kIndex = torch::kInt32;
const torch::Dtype torch_kBoundaryType = torch::kInt8; // must match BoundaryConditionType in domain_structs_gpu.h


template<typename scalar_t>
std::string S4toString(const S4<scalar_t> &vec){
	std::ostringstream repr;
	repr << "I4(" << vec.x << "," << vec.y << "," << vec.z << "," << vec.w << ")";
	return repr.str();
}
const auto I4toString = S4toString<int32_t>;
const auto F4toString = S4toString<float>;

struct Domain;
struct Block;

struct CSRmatrix{
	CSRmatrix(torch::Tensor &value, torch::Tensor &index, torch::Tensor &row);
	CSRmatrix(const int32_t numValues, const int32_t numRows, const torch::Dtype type, const torch::Device device);
	CSRmatrix(const int32_t numValues, const int32_t numRows, const py::object type, const torch::Device device) : CSRmatrix(numValues, numRows, torch::python::detail::py_object_to_dtype(type), device) {};
	void Detach();
	
	torch::Tensor value;
	void setValue(torch::Tensor &tensor);
	void CreateValue();
	torch::Tensor index;
	torch::Tensor row;
	//auto dtype() {return value.dtype();};
	//auto scalar_type() {return value.scalar_type();};
	torch::Dtype getDtype() const {return value.scalar_type();};
	torch::Device getDevice() const {return value.device();};
	// copies structures, but keeps the tensor references
	std::shared_ptr<CSRmatrix> Copy() const;
	std::shared_ptr<CSRmatrix> Clone() const;
	std::shared_ptr<CSRmatrix> WithZeroValue() const;
	std::shared_ptr<CSRmatrix> toType(const torch::Dtype type) const;
	std::shared_ptr<CSRmatrix> toTypePy(const py::object type) const {return toType(torch::python::detail::py_object_to_dtype(type));};
	index_t getSize() const {return value.size(0);};
	index_t getRows() const {return row.size(0)-1;};
	std::string ToString() const;
	
	bool IsTensorChanged() const {return isTensorChanged;};
	friend struct Domain;
protected:
	bool isTensorChanged=false;
	void setTensorChanged(const bool changed) {isTensorChanged=changed;};
};

struct Boundary{
	Boundary(const BoundaryType type, const std::shared_ptr<const Domain> p_parentDomain) : type(type), wp_parentDomain(p_parentDomain) {};
	virtual ~Boundary() = default;
	virtual std::shared_ptr<Boundary> Copy() const = 0; // {TORCH_CHECK(false, "This boundary can't be copied");};
	virtual std::shared_ptr<Boundary> Clone() const = 0;
	const BoundaryType type;
	virtual std::string ToString() const;
	
	bool IsTensorChanged() const {return isTensorChanged;};
	
	std::shared_ptr<const Domain> getParentDomain() const;
	torch::Dtype getDtype() const;
	torch::Device getDevice() const;
	index_t getSpatialDims() const;
	bool hasPassiveScalar() const;
	index_t getPassiveScalarChannels() const;
	
	friend struct Block;
	friend struct Domain;
protected:
	const std::weak_ptr<const Domain> wp_parentDomain;
	bool isTensorChanged=false;
	void setTensorChanged(const bool changed) {isTensorChanged=changed;};

};


struct FixedBoundary : Boundary {
	FixedBoundary(optional<torch::Tensor> velocity, BoundaryConditionType velocityType,
				//  torch::Tensor &pressure, BoundaryConditionType pressureType,
				  optional<torch::Tensor> passiveScalar, optional<BoundaryConditionType> passiveScalarType,
				//  const index_t passiveScalarChannels,
				  optional<torch::Tensor> transform, const std::shared_ptr<const Domain> p_parentDomain);
	/*
	FixedBoundary(const I4 size, BoundaryConditionType velocityType,
				const index_t passiveScalarChannels, optional<BoundaryConditionType> passiveScalarType,
				const torch::Dtype type, const torch::Device device);
	*/
#ifdef DTOR_MSG
	~FixedBoundary();
#endif
	void DetachFwd();
	void DetachGrad();
	void Detach();
	//bool velocityDirichlet;
	torch::Tensor m_velocity;
	bool m_velocityStatic=true; // single value (vector) for the boundary, no spatial variance
	BoundaryConditionType m_velocityType;
	void setVelocity(torch::Tensor &t);
	void setVelocityType(const BoundaryConditionType velocityType);
	void CreateVelocity(const bool createStatic);
	torch::Tensor getVelocity(const bool computational) const;
	torch::Tensor getVelocityVarying(optional<I4> size) const;
	/** extend a static velocity to a varying one by broadcasting to a new tensor. no effect if already varying.
	If the FixedBoundary already has a size it silently takes precedent over the argument.*/
	void makeVelocityVarying(optional<I4> size);
	GET_DATA_PRT(Velocity, m_velocity); //getVelocityDataPtr()

	/*
	optional<torch::Tensor> m_viscosity;
	void setViscosity(const torch::Tensor &viscosity);
	bool hasViscosity() const { return m_viscosity.has_value(); };
	bool isViscosityStatic() const { return m_viscosity ? m_viscosity.value().dim()==2 : true; };
	*/
	torch::Tensor GetFluxes() const;
	
private:
	/* Currently only Neumann with dp=0 supported. */
	torch::Tensor m_pressure;
	bool m_pressureStatic;
	BoundaryConditionType m_pressureType;
	void setPressure(torch::Tensor &t);
	void setPressureType(const BoundaryConditionType pressureType);
	void CreatePressure(const bool createStatic);
	GET_DATA_PRT(Pressure, m_pressure);
public:
	
	optional<torch::Tensor> m_passiveScalar=nullopt;
	bool isPassiveScalarStatic() const { return m_passiveScalarStatic; };
	//BoundaryConditionType m_passiveScalarType;
	optional<std::vector<BoundaryConditionType>> m_passiveScalarTypes=nullopt;
	optional<torch::Tensor> m_passiveScalarTypes_tensor=nullopt;
	GET_OPTIONAL_DATA_PRT(PassiveScalarTypes, m_passiveScalarTypes_tensor);
	//bool m_passiveScalarBoundaryTypeStatic=true; // static=the same for all channels
	bool isPassiveScalarBoundaryTypeStatic() const { return !m_passiveScalarTypes.has_value() || m_passiveScalarTypes.value().size()<2; };
	void setPassiveScalarType(const BoundaryConditionType passiveScalarType);
	void setPassiveScalarType(const std::vector<BoundaryConditionType> passiveScalarTypes);
	void setPassiveScalar(torch::Tensor &t);
	void CreatePassiveScalar(const bool createStatic);
	//bool hasPassiveScalar() const { return m_passiveScalar.has_value(); };
	//index_t getPassiveScalarChannels() const { return m_passiveScalar ? m_passiveScalar.value().size(1) : 0; };
	//void clearPassiveScalar();
	GET_OPTIONAL_DATA_PRT(PassiveScalar, m_passiveScalar);
	
	optional<torch::Tensor> m_transform=nullopt;
	bool hasTransform() const { return m_transform.has_value(); };
	void setTransform(torch::Tensor &transform); //NDHWT, T=transform data. Would NTDWH be better?
	void clearTransform();
	GET_OPTIONAL_DATA_PRT(Transform, m_transform);
	
#ifdef WITH_GRAD
	torch::Tensor m_velocity_grad = torch::empty(0);
	void setVelocityGrad(torch::Tensor &t);
	void CreateVelocityGrad();
	//torch::Tensor m_pressure_grad = torch::empty(0);
	optional<torch::Tensor> m_passiveScalar_grad;
	GET_OPTIONAL_DATA_PRT(PassiveScalarGrad, m_passiveScalar_grad);
	void setPassiveScalarGrad(torch::Tensor &t);
	void CreatePassiveScalarGrad();
	
#endif //WITH_GRAD
	
	//torch::Dtype getDtype() const;
	//torch::Device getDevice() const;

	// copies structures, but keeps the tensor references
	std::shared_ptr<Boundary> Copy() const;
	std::shared_ptr<Boundary> Clone() const;
	
	//dim_t m_numDims;
	//index_t getSpatialDims() const { return m_spatialDims; };
	/** Returns the axis index for wich this boundary can be used. Based on shape/size, returns -1 if no shape has been set. */
	index_t getCompatibleAxis() const { return m_axis; };
	/*Get the spatial axis XYZ*/
	index_t getAxis(const dim_t dim) const;
	/** Check if there is any tensor with a spatial size set.*/
	bool hasSize() const;
	I4 getSizes() const;
	I4 getStrides() const;
	std::string ToString() const;
	
	friend struct Block;
	friend struct Domain;
private:
	bool m_passiveScalarStatic=true; // static=constant over spatial dimensions
	//index_t m_spatialDims=0;
	//void setSpatialDims(const index_t dims);
	optional<I4> m_size=nullopt;
	void setSizeFromTensor(torch::Tensor &t, bool channelsFist);
	void setSize(I4 size);
	index_t m_axis=-1;
	/*Get the tensor dimension NCDHW (zyx)*/
	index_t getDim(const dim_t dim) const;
	//optional<torch::Dtype> m_dtype;
	//optional<torch::Device> m_device;
	//void setDtypeDeviceFromTensor(torch::Tensor &t);
};


struct StaticDirichletBoundary : Boundary{
	StaticDirichletBoundary(torch::Tensor &slip, torch::Tensor &velocity, torch::Tensor &passiveScalar, const std::shared_ptr<const Domain> p_parentDomain);
	torch::Tensor slip; // 1 value, cpu
	torch::Tensor boundaryVelocity; // dim values, cpu
	torch::Tensor boundaryScalar;
#ifdef WITH_GRAD
	//torch::Tensor slip_grad; // 1 value, gpu
	torch::Tensor boundaryVelocity_grad = torch::empty(0); // dim values, gpu
	//void setVelocityGrad(torch::Tensor &t);
	//void CreateVelocityGrad();
	torch::Tensor boundaryScalar_grad = torch::empty(0);
	//void setPassiveScalarGrad(torch::Tensor &t);
	//void CreatePassiveScalarGrad();
#endif
	
	//torch::Tensor transform;
	//bool hasTransform=false;
	//void setTransform(torch::Tensor &transform); //NDHWT, T=transform data. Would NTDWH be better?
	//void clearTransform();
	
	torch::Dtype getDtype() const {return boundaryVelocity.scalar_type();};
	//torch::Device getDevice() const {return boundaryVelocity.device();};
	// copies structures, but keeps the tensor references
	std::shared_ptr<Boundary> Copy() const;
	std::shared_ptr<Boundary> Clone() const;
	//torch::Tensor getMaxVelocity() const;
	//torch::Tensor getFlux() const;
	index_t getSpatialDims() const;
	std::string ToString() const;
};

struct VaryingDirichletBoundary : Boundary{
	VaryingDirichletBoundary(torch::Tensor &slip, torch::Tensor &velocity, torch::Tensor &passiveScalar, const std::shared_ptr<const Domain> p_parentDomain);
	torch::Tensor slip; // 1 value, cpu
	torch::Tensor boundaryVelocity; //includes size, rank dim+1, dim channels, gpu
	void setVelocity(torch::Tensor &t);
	torch::Tensor getVelocity(const bool computational) const;
	torch::Tensor getBoundaryFlux() const;
	torch::Tensor boundaryScalar;
	void setPassiveScalar(torch::Tensor &t);
#ifdef WITH_GRAD
	//torch::Tensor slip_grad; // 1 value, gpu
	torch::Tensor boundaryVelocity_grad = torch::empty(0); // dim values, gpu
	void setVelocityGrad(torch::Tensor &t);
	void CreateVelocityGrad();
	torch::Tensor boundaryScalar_grad = torch::empty(0);
	void setPassiveScalarGrad(torch::Tensor &t);
	void CreatePassiveScalarGrad();
#endif
	
	torch::Tensor transform;
	bool hasTransform=false;
	void setTransform(torch::Tensor &transform); //NDHWT, T=transform data. Would NTDWH be better?
	void clearTransform();
	
	torch::Dtype getDtype() {return boundaryVelocity.scalar_type();};
	torch::Device getDevice() const {return boundaryVelocity.device();};
	//torch::Tensor getMaxVelocity() const;
	//torch::Tensor getFlux() const;

	// copies structures, but keeps the tensor references
	std::shared_ptr<Boundary> Copy() const;
	std::shared_ptr<Boundary> Clone() const;
	
	//dim_t numDims;
	index_t getSpatialDims() const;
	/*Get the tensor dimension NCDHW (zyx)*/
	index_t getDim(const dim_t dim) const;
	/*Get the spatial axis XYZ*/
	index_t getAxis(const dim_t dim) const;
	I4 getSizes() const;
	I4 getStrides() const;
	std::string ToString() const;
//private:
//	bool CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const std::string &name) const;
//	torch::Tensor CreateDataTensor(const index_t channels) const;
};

struct ConnectedBoundary : Boundary{
	ConnectedBoundary(std::weak_ptr<Block> wp_connectedBlock, std::vector<dim_t> &axes, const std::shared_ptr<const Domain> p_parentDomain);
	std::shared_ptr<Block> getConnectedBlock() const;
	// 0-5 -> -x,+x,-y,+y,-z,+z
	// /2 (>>1) -> x,y,z
	// %2 (&1) -> -,+
	// dim_t connectedFace;
	// dim_t connectedAxis1;
	// dim_t connectedAxis2;
	std::vector<dim_t> axes; //always starts with the connected face
	dim_t getConnectedFace() const {return axes[0]>>1;};
	dim_t getConnectedFaceSide() const {return axes[0]&1;};
	dim_t getConnectionAxis(const dim_t axis) const;
	dim_t getConnectionAxisDirection(const dim_t axis) const;
	
	// Alternative: connections for each axis, the other block's connected face is the connection given by the entry for the axis this boundary is attached to.
	//I4 connections;
	//I4 getConnectionVector();
	std::shared_ptr<Boundary> Copy() const {TORCH_CHECK(false, "ConnectedBoundary can't be copied");};
	std::shared_ptr<Boundary> Clone() const {TORCH_CHECK(false, "ConnectedBoundary can't be cloned");};
	
	//torch::Dtype getDtype() const;
	//index_t getSpatialDims() const;
	std::string ToString() const;
private:
	// to prevent cyclic references between connected blocks.
	std::weak_ptr<Block> wp_connectedBlock;
};

struct PeriodicBoundary : Boundary{
	PeriodicBoundary(const std::shared_ptr<const Domain> p_parentDomain) : Boundary(BoundaryType::PERIODIC, p_parentDomain) {};
	std::shared_ptr<Boundary> Copy() const {return std::make_shared<PeriodicBoundary>(getParentDomain());};
	std::shared_ptr<Boundary> Clone() const {return std::make_shared<PeriodicBoundary>(getParentDomain());};
};

struct Block : public std::enable_shared_from_this<Block>{
private:
public:
	/** Construct from velocity tensor. Pressure tensor will be constructed with zero if missing. Passive scalar tensor is optional.*/
	Block(optional<torch::Tensor> velocity, optional<torch::Tensor> pressure, optional<torch::Tensor> passiveScalar,
		optional<torch::Tensor> vertexCoordinates, const std::string &name, const std::shared_ptr<const Domain> p_parentDomain);
	/** Construct velocity and pressure with zero tensors given size, type, and device. */
	Block(const I4 size, const std::string &name, const std::shared_ptr<const Domain> p_parentDomain);
	//Block(const I4 size, std::string &name, const py::object type, const torch::Device device)
	//	: Block(size, name, torch::python::detail::py_object_to_dtype(type), device) {};
#ifdef DTOR_MSG
	~Block();
#endif
	// copies structures, but keeps the tensor references. does not copy connected boundaries
	std::shared_ptr<Block> Copy() const;
	std::shared_ptr<Block> Clone() const;
	
	std::shared_ptr<const Domain> getParentDomain() const;
	
	void DetachFwd();
	void DetachGrad();
	void Detach();

	const std::string name;
	
	torch::Tensor velocity;
	void setVelocity(torch::Tensor &v);
	torch::Tensor getVelocity(const bool computational) const;
	void CreateVelocity();
	// template <typename scalar_t>
	// scalar_t* getVelocityData() {return velocity.data_ptr<scalar_t>();};
	
	
	optional<torch::Tensor> m_viscosity;
	void setViscosity(const torch::Tensor &viscosity);
	bool hasViscosity() const { return m_viscosity.has_value(); };
	bool isViscosityStatic() const { return m_viscosity ? m_viscosity.value().dim()==2 : true; };
	void clearViscosity();
	GET_OPTIONAL_DATA_PRT(Viscosity, m_viscosity);
	
	optional<torch::Tensor> velocitySource;
	bool velocitySourceStatic=false;
	void setVelocitySource(torch::Tensor &vs);
	bool hasVelocitySource() const { return velocitySource.has_value(); };
	void CreateVelocitySource(const bool createStatic);
	void clearVelocitySource();
	GET_OPTIONAL_DATA_PRT(VelocitySource, velocitySource);
	
	torch::Tensor pressure;
	void setPressure(torch::Tensor &p);
	void CreatePressure();
	// template <typename scalar_t>
	// scalar_t* getPressureData() {return velocity.data_ptr<scalar_t>();};
	
	optional<torch::Tensor> passiveScalar = nullopt;
	void setPassiveScalar(torch::Tensor &s);
	void CreatePassiveScalar();
	//void clearPassiveScalar();
	bool hasPassiveScalar() const;
	index_t getPassiveScalarChannels() const;
	GET_OPTIONAL_DATA_PRT(PassiveScalar, passiveScalar);
	
	optional<torch::Tensor> m_passiveScalarViscosity = nullopt;
	bool hasPassiveScalarViscosity() const { return m_passiveScalarViscosity.has_value(); };
	//TODO: per-cell passive scalar viscosity?
	
#ifdef WITH_GRAD
	torch::Tensor velocity_grad = torch::empty(0);
	void setVelocityGrad(torch::Tensor &vg);
	void CreateVelocityGrad();

	
	optional<torch::Tensor> m_viscosity_grad;
	void setViscosityGrad(const torch::Tensor &viscosity_grad);
	void CreateViscosityGrad();
	bool hasViscosityGrad() const { return m_viscosity_grad.has_value(); };
	void clearViscosityGrad();
	GET_OPTIONAL_DATA_PRT(ViscosityGrad, m_viscosity_grad);
	
	optional<torch::Tensor> velocitySource_grad;
	void setVelocitySourceGrad(torch::Tensor &vsg);
	bool hasVelocitySourceGrad() const { return velocitySource_grad.has_value(); };
	void CreateVelocitySourceGrad();
	void clearVelocitySourceGrad();
	GET_OPTIONAL_DATA_PRT(VelocitySourceGrad, velocitySource_grad);

	torch::Tensor pressure_grad = torch::empty(0);
	void setPressureGrad(torch::Tensor &pg);
	void CreatePressureGrad();

	torch::Tensor passiveScalar_grad = torch::empty(0);
	void setPassiveScalarGrad(torch::Tensor &sg);
	void CreatePassiveScalarGrad();
	
	void CreatePassiveScalarGradOnBoundaries();
	void CreateVelocityGradOnBoundaries();
	//void CreatePressureGradOnBoundaries();
#endif
	
	bool IsTensorChanged() const;
	torch::Tensor getMaxVelocity(const bool withBounds, const bool computational) const;
	torch::Tensor getMaxVelocityMagnitude(const bool withBounds, const bool computational) const;
	
	torch::Dtype getDtype() const;
	torch::Device getDevice() const;
	
	std::vector<std::shared_ptr<Boundary>> boundaries;
	std::vector<std::shared_ptr<Boundary>> getBoundaries() const {return boundaries;};
	void setBoundary(const std::string &side, std::shared_ptr<Boundary> boundary);
	void setBoundary(const index_t index, std::shared_ptr<Boundary> boundary);
	std::shared_ptr<Boundary> getBoundary(const std::string &side) const;
	std::shared_ptr<Boundary> getBoundary(const index_t index) const;
	/** Returns a list of pairs (boundary index, boundary) containing only the fixed boundaries of this block. */
	std::vector<std::pair<index_t, std::shared_ptr<FixedBoundary>>> getFixedBoundaries() const;
	bool isAllFixedBoundariesPassiveScalarTypeStatic() const;
	
	/** Connect this Block to another with the specified faces and axes.
	 * The connection is also set on the other block.
	 * Existing Connected and PeriodicBoudnaries are closed.
	*/
	void ConnectBlock(const dim_t face1, std::shared_ptr<Block> block2, const dim_t face2, const dim_t connectedAxis1, const dim_t connectedAxis2);
	void ConnectBlock(const std::string &face1, std::shared_ptr<Block> block2, const std::string &face2, const std::string &connectedAxis1, const std::string &connectedAxis2);
	/** Connect opposite sides of this block.
	 * Existing ConnectedBoudnaries are closed.
	*/
	void MakePeriodic(const index_t axis);
	void MakePeriodic(const std::string &axis);
	void MakeAllPeriodic();
	/** Create a FixedBoundary with Dirichlet-0 velocity and scalar at the specified face.
	 * Existing Connected and PeriodicBoudnaries cause the connected side to be closed as well.
	*/
	void CloseBoundary(const index_t face);
	void CloseBoundary(const std::string &face);
	void CloseAllBoundaries();
	/** Create FixedBoundary */
	void CloseBoundary(const index_t face, optional<torch::Tensor> velocity, optional<torch::Tensor> passiveScalar);
	void CloseBoundary(const std::string & face, optional<torch::Tensor> velocity, optional<torch::Tensor> passiveScalar);
	
	bool IsUnconnectedBoundary(const index_t index) const;
	bool hasPrescribedBoundary() const;

	/** Indicates which transformation was set. 
	 * NONE: no transformation was set.
	 * COORDS: vertexCoordinates were set, cell and face transformations were generated based on this.
	 * TRANSFORM: cell transformations were set, face transformations may have been set.
	*/
	//TransformType m_transformType = TransformType::NONE;
	optional<torch::Tensor> m_vertexCoordinates;
	GET_OPTIONAL_DATA_PRT(VertexCoordinates, m_vertexCoordinates);
	bool hasVertexCoordinates() const { return m_vertexCoordinates.has_value(); };
	void setVertexCoordinates(torch::Tensor &vertexCoordinates);
	//void clearVertexCoordinates();
	torch::Tensor getCellCoordinates() const;
	
	optional<torch::Tensor> m_transform;
	GET_OPTIONAL_DATA_PRT(Transform, m_transform);
	bool hasTransform() const { return m_transform.has_value(); };
	void setTransform(torch::Tensor &transform, optional<torch::Tensor> faceTransform); //NDHWT, T=transform data.
	//void clearTransform();

	optional<torch::Tensor> m_faceTransform;
	GET_OPTIONAL_DATA_PRT(FaceTransform, m_faceTransform);
	bool hasFaceTransform() const { return m_faceTransform.has_value(); };
	//void setFaceTransform(torch::Tensor &transform); //NCDHWT, staggered layout, C=spatial dimensions, T=transform data.
	//void clearFaceTransform();
	void clearCoordsTransforms();
	
	/** Check if all transformed coordinate systems have the same orientation by checking the sign of the determinant.
	    Returns the sign if all have the same orientation, 0 otherwise. 
	    Returns 1 if no transformations are set. */
	index_t GetCoordinateOrientation() const;
	
	//dim_t numDims;
	index_t getSpatialDims() const;
	/*Get the tensor dimension NCDHW (zyx)*/
	index_t getDim(const dim_t dim) const;
	/*Get the spatial axis XYZ*/
	index_t getAxis(const dim_t dim) const;
	I4 getSizes() const;
	I4 getStrides() const;
	
	// used by Domain
	index_t ComputeCSRSize() const;
	// set by Domain
	index_t globalOffset; //offset in cells from first block start. used e.g. for CSR indices
	index_t csrOffset;

	std::string ToString() const;
	friend struct Domain;
private:
	const std::weak_ptr<const Domain> wp_parentDomain;

	bool IsValidFaceIndex(const index_t face) const;
	bool IsValidAxisIndex(const index_t axis) const;
	void CheckFaceIndex(const index_t face) const;
	void CheckAxisIndex(const index_t axis) const;
	torch::TensorOptions getValueOptions() const;
	bool CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const bool allowStatic, const std::string &name) const;
	torch::Tensor CreateDataTensor(const index_t channels) const;
	bool isTensorChanged=true;
	void setTensorChanged(const bool changed);
	/** Close the boundary that is connected to face, if there is a connection. Does nothing to the boundary at face.*/
	void CloseConnectedBoudary(const index_t face, const bool useFixed);
	/** Makes a closed boundary at face, irrespective of existing connections*/
	void MakeClosedBoundary(const index_t face);
	/** Create a FixedBoundary with at the specified face.
	 * PeriodicBoudnaries are turned into 0-FixedBoundaries.
	 * ConnectedBoudnaries are closed.
	 * If velocity or passiveScalar are not provided they are initialzed as 0.
	 * If scalarType is not provided, velocityType is used.
	*/
	void MakeFixedBoundary(const index_t face,
		optional<torch::Tensor> velocity, const BoundaryConditionType velocityType,
		optional<torch::Tensor> passiveScalar, optional<std::vector<BoundaryConditionType>> scalarType);
	torch::Tensor GetFaceTransformBoundarySlice(const index_t face) const;
	void UpdateBoundaryTransforms();
};

void ConnectBlocks(std::shared_ptr<Block> block1, const dim_t face1, std::shared_ptr<Block> block2, const dim_t face2, const dim_t connectedAxis1, const dim_t connectedAxis2);
void ConnectBlocks(std::shared_ptr<Block> block1, const std::string &face1, std::shared_ptr<Block> block2, const std::string &face2, const std::string &connectedAxis1, const std::string &connectedAxis2);


struct Domain : public std::enable_shared_from_this<Domain>{
	
	//Domain(std::string &name);
	Domain(const index_t spatialDims, torch::Tensor &viscosity, const std::string &name,
		const torch::Dtype dtype, optional<py::object> pyDtype, const torch::Device device,
		const index_t passiveScalarChannels, optional<torch::Tensor> scalarViscosity);
	Domain(const index_t spatialDims, torch::Tensor &viscosity, const std::string &name,
		const py::object dtype, const torch::Device device,
		const index_t passiveScalarChannels, optional<torch::Tensor> scalarViscosity)
		: Domain(spatialDims, viscosity, name, torch::python::detail::py_object_to_dtype(dtype), dtype, device, passiveScalarChannels, scalarViscosity){};
#ifdef DTOR_MSG
	~Domain();
#endif

	//static std::shared_ptr<Domain> Create();
	
	/** create a block from the specified tensors, add it to the Domain's blocks, and return it. */
	std::shared_ptr<Block> CreateBlock(optional<torch::Tensor> velocity, optional<torch::Tensor> pressure, optional<torch::Tensor> passiveScalar,
		optional<torch::Tensor> vertexCoordinates, const std::string &name);
	/** create a block with zero-initialized tensors of the given size, add it to the Domain's blocks, and return it. */
	std::shared_ptr<Block> CreateBlockWithSize(I4 size, std::string &name);
	/** create a block with zero-initialized tensors from the given coordinates, add it to the Domain's blocks, and return it. */
	//std::shared_ptr<Block> CreateBlockFromCoords(torch::Tensor vertexCoordinates, const std::string &name);
	void AddBlock(std::shared_ptr<Block> block);
	// void RemoveBlock(const index_t index); // this needs to carefully handle exisiting connections!
	void PrepareSolve();
	//bool NeedPointerUpdate() const;
	void UpdateDomain();
	bool IsInitialized() {return initialized;};
	
	void DetachFwd();
	void DetachGrad();
	void Detach();

	// copies structures, but keeps the tensor references
	std::shared_ptr<Domain> Copy(optional<std::string> name) const;
	std::shared_ptr<Domain> Clone(optional<std::string> name) const;
	std::shared_ptr<Domain> To(const torch::Dtype dtype, optional<std::string> name);
	std::shared_ptr<Domain> To(const torch::Device device, optional<std::string> name);
	
	index_t getNumBlocks() const {return blocks.size();};
	index_t getMaxBlockSize() const;
	std::shared_ptr<Block> getBlock(const index_t index) {return blocks.at(index);};
	std::vector<std::shared_ptr<Block>> getBlocks() {return blocks;};
	torch::Dtype getDtype() const;
	py::object getPyDtype() const;
	torch::Device getDevice() const;
	torch::TensorOptions getTensorOptions() const;
	
	torch::Tensor getMaxVelocity(const bool withBounds, const bool computational) const;
	torch::Tensor getMaxVelocityMagnitude(const bool withBounds, const bool computational) const;
	bool hasPrescribedBoundary() const;
	bool isAllFixedBoundariesPassiveScalarTypeStatic() const;
	
	bool hasVertexCoordinates() const;
	std::vector<torch::Tensor> getVertexCoordinates() const;
	/** Check if all transformed coordinate systems have the same orientation by checking the sign of the determinant.
	    Returns the sign if all have the same orientation, 0 otherwise. */
	index_t GetCoordinateOrientation() const;
	
	torch::Tensor GetGlobalFluxBalance() const;
	bool CheckGlobalFluxBalance(const double eps) const;
	
	index_t getSpatialDims() const;
	index_t getTotalSize() const;
	
	void setViscosity(torch::Tensor &viscosity);
	torch::Tensor viscosity;
	bool hasBlockViscosity() const;
	
	// optional separate viscosity for the passive scalar.
	// specified for each channel of the passive scalar.
	// if unspecified the main viscosity is used for diffusion of all passive scalar channels.
	optional<torch::Tensor> passiveScalarViscosity;
	GET_OPTIONAL_DATA_PRT(PassiveScalarViscosity, passiveScalarViscosity);
	bool hasPassiveScalarViscosity() const { return passiveScalarViscosity.has_value(); };
	bool isPassiveScalarViscosityStatic() const { return passiveScalarViscosity ? passiveScalarViscosity.value().size(0)==1 : true; };
	void setScalarViscosity(torch::Tensor &viscosity);
	void clearScalarViscosity();
	bool hasPassiveScalarBlockViscosity() const;
	
	void CreateVelocityOnBlocks();
	void CreatePressureOnBlocks();
	void CreatePassiveScalarOnBlocks();
	void clearPassiveScalarOnBlocks();
	/** true iff all blocks have a passive scalar set. no additional tests. */
	bool hasPassiveScalar() const;
	index_t getPassiveScalarChannels() const;
	
	std::shared_ptr<CSRmatrix> C; //advection coefficients
	torch::Tensor A; //diagonal of C
	void setA(torch::Tensor &a);
	void CreateA();
	std::shared_ptr<CSRmatrix> P; //pressure coefficients
	
	optional<std::shared_ptr<CSRmatrix>> scalarC;
	
	torch::Tensor scalarRHS;
	void setScalarRHS(torch::Tensor &srhs);
	void CreateScalarRHS();
	torch::Tensor scalarResult;
	void setScalarResult(torch::Tensor &sr);
	void CreateScalarResult();
	
	torch::Tensor velocityRHS;
	void setVelocityRHS(torch::Tensor &vrhs);
	void CreateVelocityRHS();
	torch::Tensor velocityResult;
	void setVelocityResult(torch::Tensor &vr);
	void CreateVelocityResult();
	
	torch::Tensor pressureRHS;
	void setPressureRHS(torch::Tensor &prhs);
	void CreatePressureRHS();
	torch::Tensor pressureRHSdiv;
	void setPressureRHSdiv(torch::Tensor &prhsd);
	void CreatePressureRHSdiv();
	torch::Tensor pressureResult;
	void setPressureResult(torch::Tensor &pr);
	void CreatePressureResult();
	
#ifdef WITH_GRAD
	void CreatePassiveScalarGradOnBlocks();
	void CreateVelocityGradOnBlocks();
	void CreateVelocitySourceGradOnBlocks();
	void CreatePressureGradOnBlocks();
	
	void CreatePassiveScalarGradOnBoundaries();
	void CreateVelocityGradOnBoundaries();
	//void CreatePressureGradOnBoundaries();
	
	optional<torch::Tensor> viscosity_grad;
	GET_OPTIONAL_DATA_PRT(ViscosityGrad, viscosity_grad);
	bool hasViscosityGrad() const { return viscosity_grad.has_value(); };
	void setViscosityGrad(torch::Tensor &srhsg);
	void clearViscosityGrad();
	void CreateViscosityGrad();
	
	optional<torch::Tensor> passiveScalarViscosity_grad;
	GET_OPTIONAL_DATA_PRT(PassiveScalarViscosityGrad, passiveScalarViscosity_grad);
	bool hasPassiveScalarViscosityGrad() const { return passiveScalarViscosity_grad.has_value(); };
	void setPassiveScalarViscosityGrad(torch::Tensor &passiveScalarViscosity_grad);
	void clearPassiveScalarViscosityGrad();
	void CreatePassiveScalarViscosityGrad();
	
	std::shared_ptr<CSRmatrix> C_grad; //advection coefficients
	//void setCGrad(std::shared_ptr<CSRmatrix> csr);
	//void CreateCGrad();
	torch::Tensor A_grad = torch::empty(0); //diagonal of C
	void setAGrad(torch::Tensor &tensor);
	void CreateAGrad();
	std::shared_ptr<CSRmatrix> P_grad;
	//void setPGrad(std::shared_ptr<CSRmatrix> csr);
	//void CreatePGrad();
	
	torch::Tensor scalarRHS_grad = torch::empty(0);
	void setScalarRHSGrad(torch::Tensor &srhsg);
	void CreateScalarRHSGrad();
	torch::Tensor scalarResult_grad = torch::empty(0);
	void setScalarResultGrad(torch::Tensor &srg);
	void CreateScalarResultGrad();
	
	torch::Tensor velocityRHS_grad = torch::empty(0);
	void setVelocityRHSGrad(torch::Tensor &vrhs);
	void CreateVelocityRHSGrad();
	torch::Tensor velocityResult_grad = torch::empty(0);
	void setVelocityResultGrad(torch::Tensor &vr);
	void CreateVelocityResultGrad();
	
	torch::Tensor pressureRHS_grad = torch::empty(0);
	void setPressureRHSGrad(torch::Tensor &prhs);
	void CreatePressureRHSGrad();
	torch::Tensor pressureRHSdiv_grad = torch::empty(0);
	void setPressureRHSdivGrad(torch::Tensor &prhsd);
	void CreatePressureRHSdivGrad();
	torch::Tensor pressureResult_grad = torch::empty(0);
	void setPressureResultGrad(torch::Tensor &pr);
	void CreatePressureResultGrad();
#endif
	
	torch::TensorOptions getValueOptions() const { return valueOptions; };
	bool IsTensorChanged() const;
	
	//std::variant<DomainGPU<float>, DomainGPU<double>> domainGPU;
	torch::Tensor domainCPU;
	torch::Tensor domainGPU;
	DomainAtlasSet atlas;
	
	std::string ToString() const;
	std::vector<std::shared_ptr<Block>> blocks;
	
	const std::string name;
	optional<py::object> pyDtype;
	
private:
	const index_t m_spatialDims;
	const index_t m_passiveScalarChannels;
	const torch::Dtype m_dtype;
	torch::Device m_device;
	bool initialized;
	
	torch::TensorOptions valueOptions;
	bool CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const std::string &name);
	index_t totalSize;
	template <typename scalar_t>
	void SetupDomainGPU();
	template <typename scalar_t>
	void UpdateDomainGPU();
	template <typename scalar_t>
	void SetViscosityGPU();
	index_t getBlockIdx(const std::shared_ptr<const Block> block) const;
	bool isTensorChanged=true;
	void setTensorChanged(const bool changed);
};

#endif //_INCLUDE_DOMAIN_STRUCTS