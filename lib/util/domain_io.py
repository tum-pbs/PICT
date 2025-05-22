import numpy as np
import json, warnings
import torch
import PISOtorch

def dtype_to_string(dtype):
    if dtype==torch.float32:
        return "float32"
    elif dtype==torch.float64:
        return "float64"
    else:
        raise TypeError("Unsupported dtype.")

def boundary_condition_type_to_string(bound_type:PISOtorch.BoundaryConditionType):
    if bound_type==PISOtorch.BoundaryConditionType.DIRICHLET:
        return "DIRICHLET"
    elif bound_type==PISOtorch.BoundaryConditionType.NEUMANN:
        return "NEUMANN"
    else:
        raise TypeError("Unsupported boundary condition type.")

def boundary_condition_string_to_type(bound_type:PISOtorch.BoundaryConditionType):
    if bound_type=="DIRICHLET":
        return PISOtorch.BoundaryConditionType.DIRICHLET
    elif bound_type=="NEUMANN":
        return PISOtorch.BoundaryConditionType.NEUMANN
    else:
        raise TypeError("Unsupported boundary condition type.")

def bound_idx_to_str(idx):
    if idx==0:
        return '-x'
    elif idx==1:
        return '+x'
    elif idx==2:
        return '-y'
    elif idx==3:
        return '+y'
    elif idx==4:
        return '-z'
    elif idx==5:
        return '+z'

def save_domain(domain, path):
    # path without extension
    
    # tensors are saved flat
    data = []
    def add_data(tensor, dict, name):
        nonlocal data
        # handle shared tensors
        data_idx = -1
        for i, t in enumerate(data):
            if t is tensor:
                data_idx = i
                break
        if data_idx==-1:
            data_idx = len(data)
            data.append(tensor)
        dict[name] = str(data_idx)
        
    domain_dict = {}
    domain_dict["name"] = domain.name
    domain_dict["spatialDims"] = domain.getSpatialDims()
    add_data(domain.viscosity, domain_dict, "viscosity")
    domain_dict["passiveScalarChannels"] = domain.getPassiveScalarChannels()
    if domain.hasPassiveScalarViscosity():
        add_data(domain.passiveScalarViscosity, domain_dict, "passiveScalarViscosity")
    domain_dict["blocks"] = []
    
    blocks = domain.getBlocks()
    for block_idx, block in enumerate(blocks):
        block_dict = {}
        block_dict["name"] = block.name
        if block.hasViscosity():
            add_data(block.viscosity, block_dict, "viscosity")
        if block.hasPassiveScalarViscosity():
            add_data(block.passiveScalarViscosity, block_dict, "passiveScalarViscosity")
        add_data(block.velocity, block_dict, "velocity")
        add_data(block.pressure, block_dict, "pressure")
        if block.hasPassiveScalar():
            add_data(block.passiveScalar, block_dict, "scalar")
        
        if block.hasVelocitySource():
            add_data(block.velocitySource, block_dict, "velocitySource")
        
        if block.hasVertexCoordinates():
            add_data(block.vertexCoordinates, block_dict, "vertexCoordinates")
        elif block.hasTransform():
            add_data(block.transform, block_dict, "transform")
            if block.hasFaceTransform():
                add_data(block.faceTransform, block_dict, "faceTransform")
        #else:
        #    block_dict["transform"] = None
        
        block_dict["boundaries"] = []
        
        # TODO: store boundary objects flat to handle reuse
        for bound_idx in range(block.getSpatialDims()*2):
            bound = block.getBoundary(bound_idx)
            bound_dict = {}
            #bound_dict["type"] = bound.type
            if bound.type==PISOtorch.DIRICHLET:
                bound_dict["type"] = "DIRICHLET"
                add_data(bound.slip, bound_dict, "slip")
                add_data(bound.boundaryVelocity, bound_dict, "velocity")
                add_data(bound.boundaryScalar, bound_dict, "scalar")
            elif bound.type==PISOtorch.DIRICHLET_VARYING:
                bound_dict["type"] = "DIRICHLET_VARYING"
                add_data(bound.slip, bound_dict, "slip")
                add_data(bound.boundaryVelocity, bound_dict, "velocity")
                add_data(bound.boundaryScalar, bound_dict, "scalar")
                if bound.hasTransform:
                    add_data(bound.transform, bound_dict, "transform")
                #else:
                #    bound_dict["transform"] = None
            elif bound.type==PISOtorch.FIXED:
                bound_dict["type"] = "FIXED"
                bound_dict["velocityType"] = boundary_condition_type_to_string(bound.velocityType)
                add_data(bound.velocity, bound_dict, "velocity")
                if bound.hasPassiveScalar():
                    bound_dict["passiveScalarType"] = [boundary_condition_type_to_string(_) for _ in bound.passiveScalarTypes]
                    add_data(bound.passiveScalar, bound_dict, "scalar")
                if bound.hasTransform():
                    add_data(bound.transform, bound_dict, "transform")
                #else:
                #    bound_dict["transform"] = None
            elif bound.type==PISOtorch.NEUMANN:
                bound_dict["type"] = "NEUMANN"
                raise NotImplementedError
            elif bound.type==PISOtorch.CONNECTED:
                bound_dict["type"] = "CONNECTED"
                bound_dict["connectedBlock"] = blocks.index(bound.getConnectedBlock())
                bound_dict["axes"] = bound.axes
            elif bound.type==PISOtorch.PERIODIC:
                bound_dict["type"] = "PERIODIC"
            else:
                raise TypeError("Unknown boundary type.")
            
            
            block_dict["boundaries"].append(bound_dict)
        
        domain_dict["blocks"].append(block_dict)
    
    data_dict = {str(i): d.detach().cpu().numpy() for i,d in enumerate(data)}
    data_info = {str(i): {"shape": d.shape,"dtype": dtype_to_string(d.dtype),"device": str(d.device.type)} for i,d in enumerate(data)}
    domain_dict["data_info"] = data_info
    np.savez_compressed(path+".npz", **data_dict)
    with open(path+".json", "w") as file:
        json.dump(domain_dict, file)

def load_domain(path, dtype=None, device=None, with_scalar=True):
    with open(path+".json", "r") as file:
        domain_dict = json.load(file)
    with np.load(path+".npz") as data_dict:
        data_info = domain_dict["data_info"]
        data = []
        for i in range(len(data_dict)):
            data.append(torch.tensor(data_dict[str(i)], device=data_info[str(i)]["device"]).to(dtype))
    
    def get_data(dict, name):
        return data[int(dict[name])] if name in dict else None

    domain = PISOtorch.Domain(domain_dict["spatialDims"], get_data(domain_dict, "viscosity"), domain_dict["name"], dtype=dtype, device=device,
        passiveScalarChannels = domain_dict.get("passiveScalarChannels", 1) if with_scalar else 0,
        scalarViscosity = get_data(domain_dict, "passiveScalarViscosity") if with_scalar else None)

    #blocks = []
    for block_idx, block_dict in enumerate(domain_dict["blocks"]):
        #block = PISOtorch.Block(get_data(block_dict, "velocity"), get_data(block_dict, "pressure"), get_data(block_dict, "scalar"), block_dict["name"])
        vertexCoordinates = get_data(block_dict, "vertexCoordinates") if "vertexCoordinates" in block_dict else None
        block = domain.CreateBlock(get_data(block_dict, "velocity"), get_data(block_dict, "pressure"), get_data(block_dict, "scalar") if with_scalar else None,
            vertexCoordinates=vertexCoordinates, name=block_dict["name"])
        
        if "viscosity" in block_dict:
            block.setViscosity(get_data(block_dict, "viscosity"))
        
        if with_scalar and "passiveScalarViscosity" in block_dict:
            block.setPassiveScalarViscosity(get_data(block_dict, "passiveScalarViscosity"))
        
        if "velocitySource" in block_dict:
            block.setVelocitySource(get_data(block_dict, "velocitySource"))
        
        if vertexCoordinates is None and "transform" in block_dict:
            ft = get_data(block_dict, "faceTransform") if "faceTransform" in block_dict else None
            block.setTransform(get_data(block_dict, "transform"), ft)
        
        #blocks.append(block)
    blocks = domain.getBlocks()
    
    # create all blocks first to handle connected blocks via indices
    for block_idx, (block_dict, block) in enumerate(zip(domain_dict["blocks"], blocks)):
        #print("block", block_idx)
        for bound_idx, bound_dict in enumerate(block_dict["boundaries"]):
            bound_type = bound_dict["type"]
            #print("bound", bound_idx, "type", bound_type)
            bound = None
            if bound_type=="DIRICHLET":
                warnings.warn("StaticDirichletBoundary is deprecated, creating FixedBoundary instead.")
                block.CloseBoundary(bound_idx, get_data(bound_dict, "velocity"), get_data(bound_dict, "scalar"))
                #bound = PISOtorch.StaticDirichletBoundary(get_data(bound_dict, "slip"), get_data(bound_dict, "velocity"), get_data(bound_dict, "scalar"))
            elif bound_type=="DIRICHLET_VARYING":
                warnings.warn("VaryingDirichletBoundary is deprecated, creating FixedBoundary instead.")
                block.CloseBoundary(bound_idx, get_data(bound_dict, "velocity"), get_data(bound_dict, "scalar"))
                #bound = PISOtorch.VaryingDirichletBoundary(get_data(bound_dict, "slip"), get_data(bound_dict, "velocity"), get_data(bound_dict, "scalar"))
                #if bound_dict["transform"] is not None:
                #    bound.setTransform(get_data(bound_dict, "transform"))
            elif bound_type=="FIXED":
                # TODO load BoundaryConditionType when multiple are supportet
                has_scalar = with_scalar and ("scalar" in bound_dict)
                block.CloseBoundary(bound_idx, get_data(bound_dict, "velocity"), get_data(bound_dict, "scalar") if has_scalar else None)
                if has_scalar:
                    if isinstance(bound_dict["passiveScalarType"], list):
                        block.getBoundary(bound_idx).setPassiveScalarType([boundary_condition_string_to_type(_) for _ in bound_dict["passiveScalarType"]])
                    else:
                        block.getBoundary(bound_idx).setPassiveScalarType(boundary_condition_string_to_type(bound_dict["passiveScalarType"]))
                del has_scalar
            elif bound_type=="NEUMANN":
                raise NotImplementedError
            elif bound_type=="CONNECTED":
                block.ConnectBlock(bound_idx_to_str(bound_idx), blocks[bound_dict["connectedBlock"]], *[bound_idx_to_str(x) for x in bound_dict["axes"]])
                #bound = PISOtorch.ConnectedBoundary(blocks[bound_dict["connectedBlock"]], bound_dict["axes"])
            elif bound_type=="PERIODIC":
                #bound = PISOtorch.PeriodicBoundary()
                block.MakePeriodic(bound_idx//2)
            else:
                raise TypeError("Unknown boundary type: "+bound_type)
            
            if bound is not None:
                block.setBoundary(bound_idx, bound)
        
        #domain.AddBlock(block)

    #domain.PrepareSolve() this allocates lots of secondary tensors (matrix, RHS, results), so leave it to the user if needed

    return domain