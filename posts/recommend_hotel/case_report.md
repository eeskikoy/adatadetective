# Recommending Best Neighbourhoods to Book Hotels

# 1. Introduction/Business Problem

A travel agency which is specialised in arranging holidays for elderly French people, want to have a solution to find out the right travel recommendation for their client's specific accommodation needs.

While the agency provides a bunch of detailed content via their website, digging tons of pages and magazines cost endless hours of their customers without making any decision to book a holiday.

The agency states the following wishes of their customers to be meet for their guaranteed pleasant stay in the touristic destinations.

- They are elderly retired French people and they are not open to the cuisines other than French Kitchen. So they would like to eat at a French restaurant or Bistro. A destination neighbourhood having more French restaurants and bistros are preferable over other neighbourhoods having less.

- They would like to go to venues such as Museums, Performing Art Venues, Theatres and Movie Theatres. A neighbourhood with more of these venues are preferable over other neighbourhoods having less

- They would like to stay in hotels where all the venues (Restaurants, museums etc) should not be away more than 15 minutes walking distance. A hotel with lesser walking distance to venues is more preferable than the ones having a longer distance.

Based on the given customer profile and customer requirements, the agency would like to provide some mechanism for their clients to shorten the time wasted in searching for their ideal places to stay in the cities they want to visit. In other words, they would like to offer their clients some tool in their quest for finding the best neighbourhoods to stay in their desired destinations.

# 2. Data

In order to solve the problem stated above, we use the data satisfying the above requirements for a specific city _**Rotterdam**_ and its neighbourhoods to provide a solution as a proof of concept.

**Neighbourhoods and Postal Codes**

Using **BeautifulSoup**, we will scrape the neighbourhoods of the city Rotterdam from this Wikipedia page: [Neighbourhoods of Rotterdam](https://nl.wikipedia.org/wiki/Lijst_van_postcodes_3000-3999_in_Nederland) In the very first execution of our code, we will persist the scraped neighbourhoods data in the CSV file _neighbourhoods.csv_ for caching purposes. Python module _geocoder_ is used to get the centre of geolocation coordinates for the given address of each neighbourhood. Here is a small segment of _neighbourhoods.csv_ to demonstrate how its contents look like.

| Neighbourhood | Neighbourhood Postal Code | Latitude | Longitude |
|---------------------------------------|---------------------------|--------------------|--------------------|
| Stadsdriehoek/Rotterdam Centrum | 3011 | 51.917658200000005 | 4.4868516 |
| Cool/Rotterdam Centrum | 3012 | 51.9194046 | 4.4757692 |
| Rotterdam Centraal/Weena | 3013 | 51.92428870000001 | 4.4692509000000005 |
| Oude Westen | 3014 | 51.91977199999999 | 4.4660891 |
...

**Exploring Venues using Foursquare API**

Using _Foursquare API_ we could be able to fetch all the relevant venues in their corresponding neighbourhoods. In the very first execution of our code, we will persist the fetched venues through Foursquare API in the CSV file _venues.csv_ for caching purposes. The venues to be fetched will be having specific categories are as follows: _'Movie Theater', 'Theater', 'Museum', 'Music Venue', 'Performing Arts Venue', 'Bistro' and 'French Restaurant'_
Here is a small segment of the contents of _venues.csv_ to demonstrate how it looks like.

| Neighbourhood | Neighbourhood Postal Code | Neighbourhood Latitude | Neighbourhood Longitude | Venue | Venue Latitude | Venue Longitude | Venue Distance | Venue Postal Code | Venue formattedAddress | Venue Category | Venue Postal Code Prefix |
|---------------------------------|---------------------------|------------------------|-------------------------|-------------------------------------------------|--------------------|--------------------|----------------|-------------------|-----------------------------------------------------------------|-------------------|--------------------------|
| Stadsdriehoek/Rotterdam Centrum | 3011 | 51.91765820000001 | 4.4868516 | H2otel | 51.9173925337387 | 4.487061131594537 | 32 | 3011 WR | ['Wijnhaven 20a', '3011 WR Rotterdam', 'Nederland'] | Hotel | 3011 |
| Stadsdriehoek/Rotterdam Centrum | 3011 | 51.91765820000001 | 4.4868516 | ibis Rotterdam City Centre | 51.91734678762638 | 4.488300942895886 | 105 | 3011 WP | ['Wijnhaven 12', '3011 WP Rotterdam', 'Nederland'] | Hotel | 3011 |
| Stadsdriehoek/Rotterdam Centrum | 3011 | 51.91765820000001 | 4.4868516 | citizenM | 51.91925901197693 | 4.490480849856352 | 306 | 3011 WZ | ['Gelderseplein 50', '3011 WZ Rotterdam', 'Nederland'] | Hotel | 3011 |


**Distances from Venues to Hotels and Google Distance Matrix API**

The trickiest part of the problem is to relate the venues according to walking distances to hotels. Therefore we will use the _Google Distance Matrix API_ to calculate the walking distances from hotels to venues based on geolocation values. In the very first execution of our code, after constructing hotel venue relations regarding walking distances, we will persist these relations in the CSV file _hotel_venue_relation.csv_ for caching purposes. Here is a small segment of the contents of _hotel_venue_relation.csv_ to demonstrate how it looks like.

| Neighbourhood | Neighbourhood Postal Code | Hotel | Hotel Latitude | Hotel Longitude | Hotel Postal Code Prefix | Hotel Postal Code | Venue | Venue Category | Venue Latitude | Venue Longitude | Venue Postal Code Prefix | Venue Postal Code | General Venue Category | Distance To Hotel | Time To Hotel |
|---------------------------------|---------------------------|--------|------------------|-------------------|--------------------------|-------------------|-------------------|-------------------|--------------------|--------------------|--------------------------|-------------------|------------------------|-------------------|---------------|
| Stadsdriehoek/Rotterdam Centrum | 3011 | H2otel | 51.9173925337387 | 4.487061131594537 | 3011 | 3011 WR | Annabel | Music Venue | 51.92540257556953 | 4.475941780747949 | 3013 | 3013 AH | Music Venue | 1508 | 1202 |
| Stadsdriehoek/Rotterdam Centrum | 3011 | H2otel | 51.9173925337387 | 4.487061131594537 | 3011 | 3011 WR | Old Dutch | French Restaurant | 51.91577576266155 | 4.4712633464381435 | 3015 | 3015EK | French Restaurant | 1294 | 986 |
| Stadsdriehoek/Rotterdam Centrum | 3011 | H2otel | 51.9173925337387 | 4.487061131594537 | 3011 | 3011 WR | Kijk-Kubus Museum | Museum | 51.920324337784194 | 4.490140632039386 | 3011 | 3011 MH | Museum | 585 | 462 |

# 3. Methodology
Based on the requirements mentioned in the Business Problem, we are not told about the best/right neighbourhoods and we can not make any distinctions between neighbourhoods regarding the requirements of the clients.
On the other hand, we are required to recommend the neighbourhoods having similar characteristics suited to the needs of the clients so that they can find out the desired vacation destination in a short period of time. In our situation, finding similarities of a given neighbourhood can be best be addressed by Unsupervised Learning algorithms by grouping data in clusters. Therefore, we will use K-Means Clustering Algorithm.

We need to prepare an aggregated data set (Pandas Dataframe) so that the characteristics of each neighbourhood can be depicted with the features inline with the customer requirements. Here is the data frame we have prepared.

#### Aggregated Data for K-Means Algorithm

| Neighbourhood | Neighbourhood Postal Code | Avg Venue Per Hotel In Given Walking Distance | Avg Walking Time To Hotel | Total Hotel Count | Total Venue Count |
|---------------------------------------|---------------------------|-----------------------------------------------|---------------------------|-------------------|-------------------|
| Afrikaanderwijk/Katendrecht | 3072 | 11.400000 | 541.000000 | 5 | 21 |
| Cool/Rotterdam Centrum | 3012 | 10.833333 | 573.833333 | 6 | 14 |
| Coolhaveneiland | 3024 | 14.000000 | 352.500000 | 2 | 18 |
| Dijkzigt | 3015 | 12.333333 | 411.000000 | 3 | 16 |
| Feijenoord/Noordereiland/Kop van Zuid | 3071 | 9.000000 | 545.000000 | 1 | 15 |
| Middelland | 3021 | 8.000000 | 611.500000 | 3 | 16 |
| Oude Westen | 3014 | 9.000000 | 595.000000 | 1 | 16 |
| Rotterdam Centraal/Weena | 3013 | 14.000000 | 529.500000 | 1 | 18 |
| Scheepvaartkwartier/Nieuwe Werk | 3016 | 8.000000 | 486.000000 | 1 | 18 |
| Stadsdriehoek/Rotterdam Centrum | 3011 | 4.000000 | 560.500000 | 6 | 12 |


# 4. Results
***
### Clustering Neighbourhoods
Based on the given features of the [aggregated data on the previous section](#aggregated-data-for-k-means-algorithm) K-Means Clustering Algorithm returned below Cluster Labels for each neighbourhood.

| Neighbourhood | Neighbourhood Postal Code |Cluster Labels |Avg Venue Per Hotel In Given Walking Distance | Avg Walking Time To Hotel | Total Hotel Count | Total Venue Count | Latitude | Longitude |
|---------------------------------------|---------------------------|----------------|----------------------------------------------|---------------------------|-------------------|-------------------|-----------|-----------|
| Afrikaanderwijk/Katendrecht | 3072 |0 |11.400000 | 541.000000 | 5 | 21 | 51.901861 | 4.484299 |
| Cool/Rotterdam Centrum | 3012 |4 |10.833333 | 573.833333 | 6 | 14 | 51.919405 | 4.475769 |
| Coolhaveneiland | 3024 |3 |14.000000 | 352.500000 | 2 | 18 | 51.906818 | 4.457810 |
| Dijkzigt | 3015 |3 |12.333333 | 411.000000 | 3 | 16 | 51.911750 | 4.468044 |
| Feijenoord/Noordereiland/Kop van Zuid | 3071 |2 |9.000000 | 545.000000 | 1 | 15 | 51.911653 | 4.505364 |
| Middelland | 3021 |2 |8.000000 | 611.500000 | 3 | 16 | 51.917383 | 4.459214 |
| Oude Westen | 3014 |2 |9.000000 | 595.000000 | 1 | 16 | 51.919772 | 4.466089 |
| Rotterdam Centraal/Weena | 3013 |5 |14.000000 | 529.500000 | 1 | 18 | 51.924289 | 4.469251 |
| Scheepvaartkwartier/Nieuwe Werk | 3016 |2 |8.000000 | 486.000000 | 1 | 18 | 51.906240 | 4.473647 |
| Stadsdriehoek/Rotterdam Centrum | 3011 |1 |4.000000 | 560.500000 | 6 | 12 | 51.917658 | 4.486852 |

***
We have used python module _folium_ to produce following visualization of neighbourhoods associated with their clustering labels on _OpenStreetMap_ ![](https://github.com/eeskikoy/Coursera_Capstone/blob/master/img/ClusteredNeighbourhoodsVisualization.png)
***

### Recommending Best Places

Given the below travel requirements of a client, we can generate an imaginary neighbourhood and get the help of K-Means clustering to find the right label associated with existing group of neighbourhoods (Those are what so-called already trained data points)

Requirements regarding the neighbourhood:

- All the venues from hotels should be in reach by walking 10 minutes
- Hotels should be surrounded with full of desired venues so that the number of venues per hotel in 10 minutes of walking distance should be around 18 ( The client would like to visit 2 venues per day for his/her 9 days of accommodation)
- The Number of hotels in the neighbourhood should be at least 8 so that she/he can have more choices of price rates.
- The Number of venues in the neighbourhood is required to be around 25

So given above features we can provide below data set for K-Means Algorithm to predict the right cluster (in other words, right Cluster Label) containing best neighbourhoods to book hotels.

| Neighbourhood | Neighbourhood Postal Code | Avg Venue Per Hotel In Given Walking Distance | Avg Walking Time To Hotel | Total Hotel Count | Total Venue Count |
|---------------------------------------|---------------------------|-----------------------------------------------|---------------------------|-------------------|-------------------|
| ? |? | 11.400000 | 541.000000 | 5 | 21 |

***
After feeding above data to K-Means, we got the predicted cluster label = 0. There is just one neighbourhood having this cluster and it is "Afrikaanderwijk/Katendrecht"
***

| Neighbourhood | Neighbourhood Postal Code |Cluster Labels |Avg Venue Per Hotel In Given Walking Distance | Avg Walking Time To Hotel | Total Hotel Count | Total Venue Count | Latitude | Longitude |
|---------------------------------------|---------------------------|----------------|----------------------------------------------|---------------------------|-------------------|-------------------|-----------|-----------|
| Afrikaanderwijk/Katendrecht | 3072 |0 |11.400000 | 541.000000 | 5 | 21 | 51.901861 | 4.484299 |

***
We can visualize this predicted best neighbourhood using _folium_ on _OpenStreetMap_ as well.
***
![](https://github.com/eeskikoy/Coursera_Capstone/blob/master/img/PredictedNeighbourhoodsVisualization.png)


# 5. Discussion

There is no doubt that the data quality plays a very major role to get right results using Data Science.
Apparently, in this Capstone Project, We have paid less attention to the authenticity of all venues fetched from Foursquare.
While we have taken some measures to eliminate some fake venues that do not have a proper address, we could also consider other quality checking actions such as the number of likes, verification status.
By eliminating not verified and not much-liked venues, we can be more confident about the reliability of the data and about our findings consequently.

Assuring an enterprise membership and /or getting advice from the Foursquare authorities can help to get quality data regarding reliability.

# 6. Conclusion

This Capstone Project is a proof of concept showing how Data Science can be used in Tourism Sector to solve the problem of never lasting searches to find out the right touristic destinations efficiently.
Solving this kind of sophisticated problems are very much convenient than we expected. Thanks to great Machine Learning/Data Processing modules of Python plus accessible Foursquare, Google Maps data through their great API infrastructure!

