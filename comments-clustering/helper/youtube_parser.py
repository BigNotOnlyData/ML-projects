from typing import List
from urllib.parse import urlparse, parse_qs

import pyyoutube


class YouTubeApi:
    VALID_HOSTNAMES = ('youtu.be', 'www.youtube.com', 'youtube.com')

    def __init__(self, youtube_api_key: str) -> None:
        """
        Инициализация парсера t
        :param youtube_api_key: ключ полученный на официальном сайте google cloud
        """
        self.api = pyyoutube.Api(api_key=youtube_api_key)

    def extract_comments_by_video(self, video_id: str, max_comments: int, order: str) -> List[str]:
        """
        Извлечение комментариев под видео
        :param video_id: id видео
        :param max_comments: максимальное число комментариев для парсинга
        :param order: порядок полученных комментариев у API YouTube
        return: список комментариев под видео
        """
        try:
            comments = self.api.get_comment_threads(video_id=video_id,
                                                    count=max_comments,
                                                    limit=max_comments,
                                                    order=order)
            return [item.snippet.topLevelComment.snippet.textDisplay for item in comments.items]
        except pyyoutube.PyYouTubeException:
            # учитываем скрытые комментарии
            return []

    def extract_comments_by_list_of_video(self, video_ids: list, max_comments: int, order: str) -> List[List[str]]:
        """
        Извлечь комментарии из каждого видео в списке
        :param video_ids: список id видео
        :param max_comments: максимальное число комментариев для парсинга
        :param order: порядок полученных комментариев у API YouTube
        return: список списков комментариев под видео
        """
        return [self.extract_comments_by_video(video_id, max_comments, order) for video_id in video_ids]

    def get_channel_id_by_video(self, video_id: str) -> str:
        """
        Получить id канала, которому принадлежит видео
        :param video_id: id видео у которого извлекаем канал
        :return: id канала
        """
        video_by_id = self.api.get_video_by_id(video_id=video_id)
        return video_by_id.items[0].snippet.channelId

    def get_video_id_by_channel(self, channel_id: str, n_videos: int, video_id: str) -> List[str]:
        """
        Получить список из id видео-роликов с переданного канала
        :param channel_id: id канала из которнго брать видео
        :param n_videos: максимальное количество роликов
        :param video_id: id ролика, который включить в итоговый результат
        :return: список из id видео-роликов
        """

        # начальные условия
        video_ids = [video_id]
        page_token = None
        n = len(video_ids)

        while n < n_videos:
            # получаем активности на канале
            activities = self.api.get_activities_by_channel(channel_id=channel_id,
                                                            page_token=page_token,
                                                            count=n_videos,
                                                            limit=n_videos,
                                                            )
            # вычленяем id видео из активностей связанные с загрузкой видео
            list_video = [item.contentDetails.upload.videoId for item in activities.items
                          if item.snippet.type == 'upload']
            # избегаем повторного использования id
            list_video = [v for v in list_video if v != video_id]
            # обновляем начальные условия
            video_ids.extend(list_video)
            n, page_token = len(video_ids), activities.nextPageToken

            # если следущие страницы закончились выходим из цикла
            if not page_token:
                break
                
        # дополнительно фильтруем количество йдишников
        return video_ids[:n_videos]

    def extract_video_id_from_url(self, url: str) -> str:
        """
        Извлечение id из URL и проверка на валидность URL
        :param url: проверяемый url
        :return: id видео из URL
        """
        if not url.startswith(('https://', 'http://')):  
            url = 'https://'+ url

        p = urlparse(url)
        hostname = p.hostname
        if hostname not in self.VALID_HOSTNAMES:
            raise ValueError('Неверное имя хоста')

        if hostname == 'youtu.be':
            video_id = p.path[1:]
        elif hostname in ('www.youtube.com', 'youtube.com'):
            video_id = parse_qs(p.query).get('v')
            if not video_id:
                raise ValueError('Id видео не определено')
            video_id = video_id[0]
        return video_id 

    def get_comments(self, url: str, max_comments: int = 100, n_videos: int = 0,
                     order_comments: str = 'relevance') -> List[str]:
        """
        Главная функция по извлечению комментариев.
        :param url: ссылка на видео
        :param max_comments: лимит по количеству комментариев
        :param n_videos: количество дополнительных видео с того же канала, что и видео, переданное по ссылке.
               n_videos=0 - если учитывать только видео по ссылке.
        :param order_comments: порядок полученных комменатриев через API
        :return: список комменатриев
        """
        # получаем id видео
        video_id = self.extract_video_id_from_url(url)
        # если надо взять дополнительные видео с канала
        if n_videos:
            # Учитывает текущее переданное видео
            n_videos += 1
            # Получаем id канала
            channel_id = self.get_channel_id_by_video(video_id)
            # получаем список дополнительных видео
            video_ids = self.get_video_id_by_channel(channel_id=channel_id,
                                                     n_videos=n_videos,
                                                     video_id=video_id,
                                                     )
            # извлекаем комменарии
            comments = self.extract_comments_by_list_of_video(video_ids=video_ids,
                                                              max_comments=max_comments,
                                                              order=order_comments,
                                                              )
            return sum(comments, [])

        else:
            return self.extract_comments_by_video(video_id, max_comments, order_comments)
