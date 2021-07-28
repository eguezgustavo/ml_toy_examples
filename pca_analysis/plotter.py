from matplotlib import pyplot


def scree_plot(pca):
    principal_components = [pc_number + 1 for pc_number in range(len(pca.explained_variance_ratio_))]
    explained_variance = [variance *100. for variance in pca.explained_variance_ratio_]
    
    pyplot.rcParams.update({'font.size': 6})
    figure, ax = pyplot.subplots(figsize=(3, 2), dpi=200)

    ax.set_title('Scree Plot')
    ax.set_xticks(principal_components)
    ax.set_xlabel('Principal components')
    ax.set_ylabel('Explained variance (%)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.bar(principal_components, explained_variance)
    
    for index, explained_variance_value in enumerate(explained_variance):
        ax.text(x=index + 0.8 , y = explained_variance_value + 1 , s=f"{explained_variance_value:.2f}%")


def plot_3d(data):
    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2])
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    pyplot.show()


def plot_first_two_principal_components(principal_components):
    pyplot.rcParams.update({'font.size': 6})
    pyplot.figure(figsize=(2, 2), dpi=200)
    pyplot.scatter(principal_components[:,0], principal_components[:,1], s=1)

    pyplot.title('Most relevant principal components')
    pyplot.xlabel('Principal component 1')
    pyplot.ylabel('Principal component 2')
