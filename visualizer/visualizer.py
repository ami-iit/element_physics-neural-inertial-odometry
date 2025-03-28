import idyntree.bindings as idyntree
import numpy as np

class HumanURDFVisualizer:
    def __init__(self, path, model_names, 
                 base_link="Pelvis", 
                 color_palette="meshcat",
                 force_scale_factor=0.001):
        r"""
        Visualize user specified number (max 3) of human URDFs based on idyntree visualizer.
        Args:
            :param path --> to load the human URDF file
            :param model_names --> [model1, model2, ...]
        """
        self.model_names = model_names
        self.n_model = len(model_names)

        self.base_link = base_link
        self.urdf_path = path

        self.visualizer = idyntree.Visualizer()
        super().__init__
        visualizer_options = idyntree.VisualizerOptions()
        self.visualizer.init(visualizer_options)
        self.visualizer.setColorPalette(color_palette)
        self.force_scale_factor = force_scale_factor


    def load_model(self, colors):
        # ensure colors is a list of tthe correct length
        if len(colors) != self.n_model:
            raise ValueError(f"Expected number of models {self.n_model}, got numbder of colors {len(colors)}")
        
        model_loaders = []
        # initialzie the model loaders dynamically
        for _ in range(self.n_model):
            loader = idyntree.ModelLoader()
            loader.loadModelFromFile(self.urdf_path, "urdf")
            model_loaders.append(loader)
        
        # set default base frame for each model
        for loader in model_loaders:
            base_link_index = loader.model().getLinkIndex(self.base_link)
            loader.model().setDefaultBaseLink(base_link_index)
        # verify and log the base frame
        base_index = model_loaders[0].model().getDefaultBaseLink()
        base_name = model_loaders[0].model().getLinkName(base_index)
        print(f"Base frame is:; {base_name}")

        # add models to the visualzier
        for loader, name in zip(model_loaders, self.model_names):
            self.visualizer.addModel(loader.model(), name)

        # set model colors dynamically
        for name, color in zip(self.model_names, colors):
            if color is not None:
                self.visualizer.modelViz(name).setModelColor(
                    idyntree.ColorViz(idyntree.Vector4(color))
                )
        print(f"Successsfully loaded {self.n_model} models!")

    def set_camera(self, root, target):
        r"""Update the camera pose"""
        self.visualizer.camera().setPosition(root)
        self.visualizer.camera().setTarget(target)

    def update(self, s_list, H_B_list, fix_camera, camera_offset):
        r"""
        Update the models with respecttive joint positions (s) and base poses (H).
        """
        # validate inputs
        if len(s_list) != len(H_B_list):
            raise ValueError(f"s_list and H_B_list must have the same length!")
        
        s_idyntree_list = [idyntree.VectorDynSize(s) for s in s_list]
        #print(f"step:{step}, Joint positions: {s_idyntree_list[0]}")
        T_b_list = [idyntree.Transform() for _ in range(len(H_B_list))]
        for i, H_B in enumerate(H_B_list):
            T_b_list[i].fromHomogeneousTransform(idyntree.Matrix4x4(H_B))
        
        # update the camera pose 
        if fix_camera:
            camera_pos_root = T_b_list[0].getPosition() + idyntree.Position(np.array(camera_offset))
            camera_pos_target = T_b_list[0].getPosition()
            self.set_camera(camera_pos_root, camera_pos_target)
        else:
            self.visualizer.camera().animator().enableMouseControl()

        # update the models
        for i, name in enumerate(self.model_names):
            self.visualizer.modelViz(name).setPositions(T_b_list[i], s_idyntree_list[i])

    def run(self):
        self.visualizer.draw()

        

