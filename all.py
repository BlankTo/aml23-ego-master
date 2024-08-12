from save_feat import main
from utils.args import args
from cluster_analysis import analyze_clusters

if __name__ == "__main__":
    main()

    analyze_clusters(args.name, [4, 8, 16, 64])

## call with: c:/Users/User/Desktop/aml_project/aml23-ego-master/aml_venv/Scripts/python.exe c:/Users/User/Desktop/aml_project/aml23-ego-master/all.py config=configs/I3D_save_feat.yaml dataset.shift=D1-D1 dataset.RGB.data_path=ek_data\frames n_frame=25