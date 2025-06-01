# python3.7
"""Model zoo."""

# pylint: disable=line-too-long

# limited to StyleGAN

MODEL_ZOO = {
    # StyleGAN official.
    'stylegan_ffhq1024': dict(
        gan_type='stylegan',
        resolution=1024,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EdfMxgb0hU9BoXwiR3dqYDEBowCSEF1IcsW3n4kwfoZ9OQ?e=VwIV58&download=1',
    ),
    'stylegan_celebahq1024': dict(
        gan_type='stylegan',
        resolution=1024,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcCdXHddE7FOvyfmqeOyc9ABqVuWh8PQYFnV6JM1CXvFig?e=1nUYZ5&download=1',
    ),
    'stylegan_bedroom256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ea6RBPddjcRNoFMXm8AyEBcBUHdlRNtjtclNKFe89amjBw?e=Og8Vff&download=1',
    ),
    'stylegan_cat256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EVjX8u9HuehLip3z0hRfIHcB7QtoFkTB7NiRDb8nrKOl2w?e=lHcp1B&download=1',
    ),
    'stylegan_car512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcRJNNzzUzJGjI2X53S9HjkBhXkKT5JRd6Q3IIhCY1AyRw?e=FvMRNj&download=1',
    ),

    # StyleGAN ours.
    'stylegan_celeba_partial256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ET2etKNzMS9JmHj5j60fqMcBRJfQfYNvqUrujaIXxCvKDQ?e=QReLE6&download=1',
    ),
    'stylegan_ffhq256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ES-NAUCC2qdHg87BftvlBiQBVpbJ8-005Q4TNr5KrOxQEw?e=00AnWt&download=1',
    ),
    'stylegan_ffhq512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZYrrwOiEgVOg-PfGv7QTegBzFQ9yq2v7o1WxNq5JJ9KNA?e=SZU8PI&download=1',
    ),
    'stylegan_livingroom256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfFCYLHjqbFDmjOvCCFJgDcBZ1QYgETfZJxp4ZTHjLxZBg?e=InVd0n&download=1',
    ),
    'stylegan_diningroom256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERsUza_hSFRIm4iZCag7P0kBQ9EIdfQKByw4QYt_ay97lg?e=Cimh7S&download=1',
    ),
    'stylegan_kitchen256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERcYvoingQNKix35lUs0vUkBQQkAZMp1rtDxjwNlOJAoaA?e=a1Tcwr&download=1',
    ),
    'stylegan_apartment256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfurPNSB2BRFtXdqGkmDD6YBwyKN8YK2v7nKwnJQdsbf6A?e=w3oYa4&download=1',
    ),
    'stylegan_church256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETMgG1_d06tAlbUkJD1qA9IBaLZ9zJKPkG2kO-4jxhVV5w?e=Dbkb7o&download=1',
    ),
    'stylegan_tower256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ebm9QMgqB2VDqyIE5rFhreEBgZ_RyKcRf8bQ333K453u3w?e=if8sDj&download=1',
    ),
    'stylegan_bridge256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ed9QM6OP9sVHnazSp4cqPSEBb-ALfBPXRxP1hD7FsTYh8w?e=3vv06p&download=1',
    ),
    'stylegan_restaurant256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ESDhYr01WtlEvBNFrVpFezcB2l9lF1rBYuHFoeNpBr5B7A?e=uFWFNh&download=1',
    ),
    'stylegan_classroom256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbWnI3oto9NPk-lxwZlWqPQB2atWpGiTWMIT59MzF9ij9Q?e=KvcNBg&download=1',
    ),
    'stylegan_conferenceroom256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb1gVi3pGa9PgJ4XYYu_6yABQZ0ZcGDak4FEHaTHaeYFzw?e=0BeE8t&download=1',
    ),

    # StyleGAN third-party.
    'stylegan_animeface512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWDWflY6lBpGgX0CGQpd2Z4B5wTEVamTOA9JRYne7zdCvA?e=tOzgYA&download=1',
    ),
    'stylegan_animeportrait512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXBvhTBi-v5NsnQtrxhFEKsBin4xg-Dud9Jr62AEwFTIxg?e=bMGK7r&download=1',
    ),
    'stylegan_artface512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eca0OiGqhyZMmoPbKahSBWQBWvcAH4q2CE3zdZJflp2jkQ?e=h4rWAm&download=1',
    )
}

# pylint: enable=line-too-long
