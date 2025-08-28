from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        # Stores the representative color of each team
        self.team_colors = {}
        # Stores which player_id is assigned to which team
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        """
        Runs KMeans clustering on an image to find 2 dominant colors.
        """
        # Reshape the image to a 2D array of pixels (num_pixels, 3 for RGB)
        image_2d = image.reshape(-1, 3)

        # Perform KMeans with 2 clusters (jersey vs background/other color)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant jersey color for a single player given their bounding box.
        """
        # Crop player image from the frame using bounding box coordinates
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Only take the top half of the player (usually jersey area)
        top_half_image = image[0:int(image.shape[0]/2), :]

        # Run clustering on the top half
        kmeans = self.get_clustering_model(top_half_image)

        # Cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels back to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Look at the corner pixels → likely background
        corner_clusters = [
            clustered_image[0, 0],         # top-left corner
            clustered_image[0, -1],        # top-right corner
            clustered_image[-1, 0],        # bottom-left corner
            clustered_image[-1, -1]        # bottom-right corner
        ]

        # Find the most common cluster among corners → background
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Player jersey cluster is the opposite one
        player_cluster = 1 - non_player_cluster

        # Extract jersey RGB color from cluster centers
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Determines the team colors by clustering all players' jersey colors.
        """
        player_colors = []

        # For each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            # Get that player’s jersey color
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Cluster all players into 2 teams based on jersey color
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Save clustering model for later team assignment
        self.kmeans = kmeans

        # Store team representative colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Assigns a player to a team based on their jersey color.
        """
        # If player already has a team assigned, return it directly
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Otherwise, get jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict which team this color belongs to
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        # Convert from {0,1} to {1,2}
        team_id += 1

        # Special case: force player 91 into team 1 (custom rule)
        if player_id == 91:
            team_id = 1

        # Save assignment
        self.player_team_dict[player_id] = team_id

        return team_id
